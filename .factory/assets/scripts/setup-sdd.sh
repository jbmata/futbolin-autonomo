#!/usr/bin/env bash
# Software Factory — install the chosen SDD engine (gsd | spec-kit | openspec) for every runtime present
# (Claude Code, GitHub Copilot CLI, opencode). The engine is a PROJECT DECISION: /factory-init explains the
# three engines, the user chooses, then the agent runs this helper. It never picks a default.
# Engine-specific options come from .factory/config.yml (sdd.<engine>.*); tokens for private sources from
# .factory/sdd/secrets.env (git-ignored). Idempotent and best-effort.
#
# Usage:   setup-sdd.sh [gsd|spec-kit|openspec] [--migrate] [--install-node] [TARGET_DIR]
#          (engine defaults to sdd.engine in config.yml)
#          gsd asks before two changes outside the repo: installing Node 24 (--install-node) and replacing an old
#          get-shit-done-cc install (--migrate). On a terminal it asks; run by an agent it prints the plan and exits 3,
#          and the agent re-runs it with the flag once the user confirms.
# Secrets: .factory/sdd/secrets.env → GSD_GIT_TOKEN, SPECKIT_CATALOG_TOKEN
# Exit:    0 done (best-effort) | 2 usage or no engine | 3 waiting for the user's confirmation
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
. "$SCRIPT_DIR/_lib.sh"

ENGINE=""; TARGET=""; MIGRATE=0; INSTALL_NODE=0; PENDING=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    gsd|spec-kit|openspec) ENGINE="$1"; shift ;;
    --migrate)      MIGRATE=1; shift ;;
    --install-node) INSTALL_NODE=1; shift ;;
    -*) say "Unknown flag: $1   (usage: setup-sdd.sh [gsd|spec-kit|openspec] [--migrate] [--install-node] [TARGET_DIR])"; exit 2 ;;
    *)  if [ -d "$1" ]; then TARGET="$1"; shift
        else say "Unknown SDD engine or directory: '$1'   (engines: gsd | spec-kit | openspec)"; exit 2; fi ;;
  esac
done
TARGET="$(resolve_target "$TARGET")" || { say "Target dir not found"; exit 1; }
CONFIG_FILE="$TARGET/.factory/config.yml"

[ -z "$ENGINE" ] && [ -f "$CONFIG_FILE" ] && ENGINE="$(cfg_get sdd.engine)"
if [ -z "$ENGINE" ]; then
  say "No SDD engine chosen. Choose it in /factory-init (the agent explains gsd, spec-kit and openspec),"
  say "or run: setup-sdd.sh <gsd|spec-kit|openspec> [TARGET_DIR]"
  exit 2
fi
case "$ENGINE" in
  gsd|spec-kit|openspec) ;;
  *) say "Unknown SDD engine '$ENGINE' (options: gsd | spec-kit | openspec)"; exit 2 ;;
esac

# ── Load secrets (best-effort; never echo values) ──────────────────────────────
SECRETS_FILE="$TARGET/.factory/sdd/secrets.env"
if [ -f "$SECRETS_FILE" ]; then
  set +u; . "$SECRETS_FILE"; set -u
fi
GSD_GIT_TOKEN="${GSD_GIT_TOKEN:-}"
SPECKIT_CATALOG_TOKEN="${SPECKIT_CATALOG_TOKEN:-}"

# ── Engine options from config.yml ─────────────────────────────────────────────
GSD_SOURCE=""; SPECKIT_SOURCE=""; SPECKIT_CATALOG_URL_CFG=""; OPENSPEC_SOURCE=""; OPENSPEC_TOOLS=""; OPENSPEC_PROFILE=""
if [ -f "$CONFIG_FILE" ]; then
  GSD_SOURCE="$(cfg_get sdd.gsd.source)"
  SPECKIT_SOURCE="$(cfg_get sdd.speckit.source)"
  SPECKIT_CATALOG_URL_CFG="$(cfg_get sdd.speckit.catalog_url)"
  OPENSPEC_SOURCE="$(cfg_get sdd.openspec.source)"
  OPENSPEC_TOOLS="$(cfg_get sdd.openspec.tools)"
  OPENSPEC_PROFILE="$(cfg_get sdd.openspec.profile)"
fi
# Env var wins over config for the catalog URL
SPECKIT_CATALOG_URL="${SPECKIT_CATALOG_URL:-${SPECKIT_CATALOG_URL_CFG:-}}"

detect_runtimes
say "════════════════════════════════════════════════════════"
say " Software Factory — SDD engine: $ENGINE"
say " repo:     $TARGET"
say " runtimes: ${RUNTIMES:-none detected (installing for Claude Code)}"
say "════════════════════════════════════════════════════════"

# ── Helper: install_gsd ────────────────────────────────────────────────────────
# GSD Core (@opengsd/gsd-core, successor of get-shit-done-cc), pinned to sdd.gsd.version — never @latest. One
# installer run takes several runtime flags (--claude --opencode --copilot), one per entry of runtime_targets.
# (`--all` would install all 18 runtimes GSD Core supports, so it is never used.)
gsd_dir(){   # config dir GSD writes for a runtime
  case "$1" in
    claude)   printf '%s' "${CLAUDE_CONFIG_DIR:-$HOME/.claude}" ;;
    opencode) printf '%s' "${OPENCODE_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/opencode}" ;;
    copilot)  printf '%s' "${COPILOT_CONFIG_DIR:-${COPILOT_HOME:-$HOME/.copilot}}" ;;
  esac
}

# confirm <flag 0|1> <question> — yes when the flag was passed; on a terminal, ask; otherwise (an agent runs the
# script) answer no and mark the run PENDING, so the agent asks the user and re-runs with the flag.
confirm(){
  local a
  [ "$1" = 1 ] && return 0
  if [ -t 0 ]; then printf '%s [s/N] ' "$2" >/dev/tty; read -r a </dev/tty; [[ "$a" =~ ^[sSyY] ]]; return; fi
  PENDING=1; return 1
}

# GSD Core needs Node >= 24, and Claude Code runs the GSD hooks with the node active when it starts, so 24 becomes
# the default for new shells. User level only (fnm or nvm, no sudo); the current Node stays installed.
ensure_node24(){
  local major nvm_sh="${NVM_DIR:-$HOME/.nvm}/nvm.sh" plan
  major="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
  [ "$major" -ge 24 ] && { say "  ✓ node $(node --version)"; return 0; }
  if have fnm; then plan="fnm install 24 && fnm default 24"
  elif [ -s "$nvm_sh" ]; then plan="nvm install 24 && nvm alias default 24"
  else plan="install nvm v0.40.3 in ~/.nvm, then nvm install 24 && nvm alias default 24"; fi
  say "  ⚠ GSD Core needs Node >= 24 (active: $(node --version 2>/dev/null || echo none))."
  say "    Plan: $plan. The current Node stays installed; 24 becomes the default for new shells."
  confirm "$INSTALL_NODE" "  Install Node 24 now?" \
    || { say "  → once the user confirms: setup-sdd.sh gsd --install-node $TARGET"; return 1; }
  if have fnm; then
    fnm install 24 && fnm default 24 && eval "$(fnm env --shell bash)" && fnm use 24 >/dev/null || return 1
  else
    if [ ! -s "$nvm_sh" ]; then
      curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash || return 1
    fi
    set +u; . "$nvm_sh" && nvm install 24 && nvm alias default 24 && nvm use 24 >/dev/null; local rc=$?; set -u
    [ "$rc" = 0 ] || return 1
  fi
  say "  ✓ node $(node --version) — restart the terminal and Claude Code so they start with Node 24"
}

# migrate_legacy <runtime flags> — get-shit-done-cc → GSD Core. The old package goes first, through its own
# uninstaller (run after GSD Core it would delete the new skills/gsd-*, which share the names). Shows both plans and
# applies them only after confirmation. Removes nothing outside what GSD manages; .planning/ is never touched.
migrate_legacy(){
  local rt d ver found=""
  # An install has get-shit-done/VERSION; the uninstaller removes it (a kept USER-PROFILE.md is only a leftover).
  for rt in claude opencode copilot; do [ -f "$(gsd_dir "$rt")/get-shit-done/VERSION" ] && found="$found $rt"; done
  [ -z "$found" ] && return 0
  say "  get-shit-done-cc installed in:$found — GSD Core replaces it. What would be removed:"
  say "  · GSD Core legacy cleanup (gsd-core --dry-run):"
  # shellcheck disable=SC2086
  npx -y --package="$GSD_PKG" -- gsd-core $1 --global --dry-run </dev/null 2>&1 \
    | grep -E '\[dry-run\]|\[legacy-cleanup\]|no legacy' | sed 's/^/      /'
  for rt in $found; do
    d="$(gsd_dir "$rt")"; ver="$(cat "$d/get-shit-done/VERSION")"
    say "  · get-shit-done-cc@$ver --$rt --global --uninstall, in $d: skills/gsd-* ($(ls -d "$d"/skills/gsd-* 2>/dev/null | wc -l)),"
    say "      agents/gsd-*, hooks/gsd-*, commands/gsd/, get-shit-done/ (keeps USER-PROFILE.md), its hooks and"
    say "      statusline in settings.json, gsd-file-manifest.json. Nothing else."
  done
  confirm "$MIGRATE" "  Uninstall get-shit-done-cc and install GSD Core $GSD_VERSION?" \
    || { say "  → once the user confirms: setup-sdd.sh gsd --migrate $TARGET"; return 1; }
  for rt in $found; do
    ver="$(cat "$(gsd_dir "$rt")/get-shit-done/VERSION")"   # its own version's uninstaller
    npx -y --package="get-shit-done-cc@$ver" -- get-shit-done-cc "--$rt" --global --uninstall </dev/null \
      || { say "  ✗ get-shit-done-cc uninstall failed ($rt) — GSD Core NOT installed; nothing else was changed"; return 1; }
  done
  say "  ✓ get-shit-done-cc uninstalled from:$found"
  need=1   # install_gsd's local: GSD Core goes in right after, whatever was there before
}

# GSD Core 1.20.1's legacy cleanup misses hooks that still name get-shit-done-cc, and nothing removes an old
# get-shit-done/ dir: report what is left, never delete it here.
report_leftovers(){
  local rt d left
  left="$(for rt in claude opencode copilot; do
    d="$(gsd_dir "$rt")"
    [ -d "$d/get-shit-done" ] && printf '%s\n' "$d/get-shit-done/"
    grep -rls "get-shit-done-cc" "$d/hooks" "$d/commands" "$d/settings.json" 2>/dev/null
  done)"
  if [ -z "$left" ]; then say "  ✓ no get-shit-done-cc leftovers"
  else say "  ⚠ get-shit-done-cc leftovers (review and remove by hand; an old USER-PROFILE.md is yours to keep):"
       printf '%s\n' "$left" | sed 's/^/      /'; fi
}

install_gsd(){
  local rt flags="" need=0
  GSD_VERSION="$(cfg_get sdd.gsd.version)"
  case "$GSD_VERSION" in
    ""|latest|next) say "  ✗ sdd.gsd.version must pin an exact GSD Core version in .factory/config.yml (never latest), e.g.:"
                    say '      sdd:'; say '        gsd:'; say '          version: "1.20.1"'; return 1 ;;
  esac
  GSD_TARGETS="$(cfg_get runtime_targets | tr -d '[],')"
  [ -z "$GSD_TARGETS" ] && GSD_TARGETS="${RUNTIMES:-claude}"
  for rt in $GSD_TARGETS; do
    case "$rt" in
      claude|opencode|copilot)
        flags="$flags --$rt"
        [ "$(cat "$(gsd_dir "$rt")/gsd-core/VERSION" 2>/dev/null)" = "$GSD_VERSION" ] || need=1 ;;
      *) say "  ○ runtime '$rt' is not a factory runtime — skipped" ;;
    esac
  done
  [ -z "$flags" ] && { say "  ✗ runtime_targets has no claude, opencode or copilot"; return 1; }

  GSD_PKG="@opengsd/gsd-core@$GSD_VERSION"
  if [ -n "$GSD_SOURCE" ]; then
    # Private / custom source: any npm package spec whose bin is gsd-core (git+ssh uses the system SSH key).
    GSD_PKG="$GSD_SOURCE"
    if [[ "$GSD_SOURCE" == git+https://* ]] && [ -n "$GSD_GIT_TOKEN" ]; then
      GSD_PKG="${GSD_SOURCE/#git+https:\/\//git+https:\/\/${GSD_GIT_TOKEN}@}"
    fi
    say "  custom source: $GSD_SOURCE (version $GSD_VERSION)"
  fi

  ensure_node24 || return 1
  have npx || { say "  ✗ npx not available — install Node 24 (npm >= 10)"; return 1; }
  migrate_legacy "$flags" || return 1
  if [ "$need" = 0 ]; then
    say "  ✓ GSD Core $GSD_VERSION already installed for $GSD_TARGETS — to change it, edit sdd.gsd.version and re-run"
    report_leftovers; return 0
  fi
  say "  installing GSD Core $GSD_VERSION (global:$flags)…"
  # shellcheck disable=SC2086
  if npx -y --package="$GSD_PKG" -- gsd-core $flags --global </dev/null >/dev/null; then
    say "  ✓ GSD Core $GSD_VERSION installed:$flags"
  else
    say "  ⚠ failed — manual: npx -y --package=${GSD_SOURCE:-@opengsd/gsd-core@$GSD_VERSION} -- gsd-core$flags --global"
    return 1
  fi
  report_leftovers
}

# ── Helper: install_speckit ────────────────────────────────────────────────────
install_speckit(){
  # Determine the specify binary
  SPECIFY_CMD=""
  if have specify; then
    SPECIFY_CMD="specify"; say "  ✓ specify CLI already present"
  elif have uvx; then
    SPECIFY_CMD="uvx specify"; say "  using uvx specify"
  elif have pipx; then
    say "  installing specify via pipx…"
    pipx install specify </dev/null && SPECIFY_CMD="specify" || true
    have specify && SPECIFY_CMD="specify"
  fi
  if [ -z "$SPECIFY_CMD" ]; then
    say "  ⚠ specify not available — install: pipx install specify  or  uvx specify"
    return 1
  fi

  # Init for EVERY runtime (spec-kit writes per-agent slash commands: .claude/commands/ for Claude Code,
  # .opencode/command/ for opencode, .github/prompts/ for GitHub Copilot). Run once per --ai target, into
  # the current dir via --here; the shared .specify/ templates are identical, so the runs are additive.
  SPECKIT_AGENTS="${SPECKIT_AGENTS:-claude opencode copilot}"
  FROM_ARG=""; [ -n "$SPECKIT_SOURCE" ] && FROM_ARG="--from $SPECKIT_SOURCE"
  [ -n "$SPECKIT_SOURCE" ] && say "  custom source: $SPECKIT_SOURCE"
  for ag in $SPECKIT_AGENTS; do
    say "  specify init (--here --ai $ag)…"
    # shellcheck disable=SC2086
    $SPECIFY_CMD init --here --ai "$ag" $FROM_ARG </dev/null \
      && say "  ✓ specify init [$ag]" \
      || say "  ⚠ specify init [$ag] best-effort — manual: $SPECIFY_CMD init --here --ai $ag $FROM_ARG"
  done

  # Private catalog: add it so extensions can be resolved
  if [ -n "$SPECKIT_CATALOG_URL" ]; then
    say "  adding private catalog: $SPECKIT_CATALOG_URL"
    if [ -n "$SPECKIT_CATALOG_TOKEN" ]; then
      SPECKIT_CATALOG_TOKEN="$SPECKIT_CATALOG_TOKEN" \
        $SPECIFY_CMD extension catalog add "$SPECKIT_CATALOG_URL" --name private </dev/null \
        && say "  ✓ private catalog registered" \
        || say "  ⚠ could not register the catalog — do it manually"
    else
      $SPECIFY_CMD extension catalog add "$SPECKIT_CATALOG_URL" --name private </dev/null \
        && say "  ✓ catalog registered" \
        || say "  ⚠ could not register the catalog — do it manually"
    fi
  fi

  # Install extensions listed under sdd.speckit.extensions[]
  if [ -f "$CONFIG_FILE" ]; then
    # Extract extension entries from the YAML block. Expects indented list entries:
    #   "    - name: foo" with optional "      from: <url>" and "      subdir: <path>".
    # A git-URL `from` is cloned into .factory/sdd/vendor/<name>/ (kept — `--dev` may reference
    # the path) and installed with `specify extension add --dev <clone>/<subdir>`; a non-URL
    # `from` uses `--from`; an empty `from` resolves <name> from the configured catalog.
    GSD_GIT_TOKEN="$GSD_GIT_TOKEN" SPECKIT_CATALOG_TOKEN="$SPECKIT_CATALOG_TOKEN" \
      SPECKIT_CATALOG_URL="$SPECKIT_CATALOG_URL" \
      python3 - "$CONFIG_FILE" "$SPECIFY_CMD" "$TARGET/.factory/sdd/vendor" <<'PY'
import os, sys, re, subprocess, shlex
config_file = sys.argv[1]
specify_cmd = sys.argv[2]
vendor_dir = sys.argv[3]

with open(config_file) as f:
    text = f.read()

# Find the speckit.extensions block (tolerate inline "# …" comments and comment-only lines —
# the config schema factory-init writes carries them)
match = re.search(r'\bspeckit:[ \t]*(?:#[^\n]*)?\n((?:[ \t]+.*\n)*)', text)
if not match:
    sys.exit(0)
block = match.group(1)
ext_section = re.search(
    r'[ \t]+extensions:[ \t]*(?:#[^\n]*)?\n((?:[ \t]+#.*\n)*(?:[ \t]+-.*\n(?:[ \t]+[^-].*\n)*)*)',
    block)
if not ext_section:
    sys.exit(0)

def val(line, key):
    # value after "key:", with inline comments (" #…") and surrounding quotes stripped
    raw = line.split(key + ':', 1)[1]
    raw = re.split(r'\s+#', raw, 1)[0].strip()
    return raw.strip('"').strip("'")

entries = []
current = {}
for line in ext_section.group(1).splitlines():
    stripped = line.strip()
    if stripped.startswith('- name:'):
        if current:
            entries.append(current)
        current = {'name': val(stripped, 'name'), 'from': '', 'subdir': ''}
    elif stripped.startswith('from:') and current:
        current['from'] = val(stripped, 'from')
    elif stripped.startswith('subdir:') and current:
        current['subdir'] = val(stripped, 'subdir')
if current:
    entries.append(current)

def is_git_url(u):
    return u.startswith(('git+', 'http://', 'https://', 'git@', 'ssh://'))

for ext in entries:
    name = ext['name']
    from_url = ext.get('from', '')
    subdir = ext.get('subdir', '')
    if not name or name == 'none':
        continue
    if from_url and is_git_url(from_url):
        # Clone (or update) the extension repo, then install its package dir with --dev.
        # The token is injected per invocation only — never persisted in the clone's .git/config.
        clone_url = from_url[4:] if from_url.startswith('git+') else from_url
        dest = os.path.join(vendor_dir, name)
        authed = clone_url
        token = os.environ.get('GSD_GIT_TOKEN', '')
        host = clone_url.split('//', 1)[1].split('/', 1)[0] if '//' in clone_url else ''
        if token and clone_url.startswith('https://') and '@' not in host:
            authed = clone_url.replace('https://', 'https://%s@' % token, 1)
        if os.path.isdir(os.path.join(dest, '.git')):
            subprocess.run(['git', '-C', dest, 'pull', '--ff-only', authed],
                           stdin=subprocess.DEVNULL, capture_output=True, text=True)
        else:
            os.makedirs(vendor_dir, exist_ok=True)
            print(f"  cloning {name} ({clone_url})...")
            result = subprocess.run(['git', 'clone', '--depth', '1', authed, dest],
                                    stdin=subprocess.DEVNULL, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ⚠ {name}: clone failed — authenticate first (az login / credential "
                      f"helper, or GSD_GIT_TOKEN in .factory/sdd/secrets.env)")
                continue
            if authed != clone_url:
                subprocess.run(['git', '-C', dest, 'remote', 'set-url', 'origin', clone_url],
                               stdin=subprocess.DEVNULL, capture_output=True, text=True)
        path = os.path.join(dest, subdir) if subdir else dest
        if not os.path.isdir(path):
            print(f"  ⚠ {name}: subdir '{subdir}' not found in the cloned repo")
            continue
        cmd = shlex.split(specify_cmd) + ['extension', 'add', '--dev', path]
        label = f"  adding extension {name} (--dev {path})..."
    elif from_url:
        cmd = shlex.split(specify_cmd) + ['extension', 'add', name, '--from', from_url]
        label = f"  adding extension {name} (from {from_url})..."
    else:
        cmd = shlex.split(specify_cmd) + ['extension', 'add', name]
        label = f"  adding extension {name}..."
    print(label)
    result = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✓ {name}")
    else:
        print(f"  ⚠ {name} failed: {result.stderr.strip()[:120]}")
PY
  fi
  say "  ✓ spec-kit ready"
}

# ── Helper: install_openspec ───────────────────────────────────────────────────
install_openspec(){
  # OpenSpec (Fission-AI) — Node CLI, no MCP/keys. Requires Node ≥ 20.19.0.
  if have openspec; then
    say "  ✓ openspec already installed ($(openspec --version 2>/dev/null | head -1)) — update the CLI with 'npm install -g @fission-ai/openspec@latest' ('openspec update' only refreshes the in-repo instructions)"
  else
    if ! have node; then
      say "  ✗ node not available — OpenSpec requires Node ≥ 20.19.0"; return 1
    fi
    if [ -n "$OPENSPEC_SOURCE" ]; then
      say "  custom source: $OPENSPEC_SOURCE"
      if [[ "$OPENSPEC_SOURCE" == git+ssh://* || "$OPENSPEC_SOURCE" == git+git@* ]]; then
        say "  installing from SSH (uses the system SSH key)…"
        npm install -g "$OPENSPEC_SOURCE" </dev/null \
          && say "  ✓ openspec custom installed" \
          || say "  ⚠ failed — check that the SSH key has access to the repository"
      elif [[ "$OPENSPEC_SOURCE" == git+https://* ]] && [ -n "$GSD_GIT_TOKEN" ]; then
        AUTHED_URL="${OPENSPEC_SOURCE/#git+https:\/\//git+https:\/\/${GSD_GIT_TOKEN}@}"
        say "  installing from HTTPS with token (GSD_GIT_TOKEN)…"
        npm install -g "$AUTHED_URL" </dev/null \
          && say "  ✓ openspec custom installed" \
          || say "  ⚠ failed — check the token in .factory/sdd/secrets.env (GSD_GIT_TOKEN)"
      else
        say "  installing: npm install -g $OPENSPEC_SOURCE"
        npm install -g "$OPENSPEC_SOURCE" </dev/null \
          && say "  ✓ openspec custom installed" \
          || say "  ⚠ failed — install manually: npm install -g $OPENSPEC_SOURCE"
      fi
    else
      say "  installing @fission-ai/openspec (public, global)…"
      npm install -g @fission-ai/openspec@latest </dev/null \
        && say "  ✓ openspec installed" \
        || say "  ⚠ failed — manual: npm install -g @fission-ai/openspec@latest"
    fi
  fi

  # openspec init for the factory runtimes (idempotent; emits skills/commands per tool)
  if have openspec; then
    OPENSPEC_TOOLS_RESOLVED="$OPENSPEC_TOOLS"
    [ -z "$OPENSPEC_TOOLS_RESOLVED" ] && OPENSPEC_TOOLS_RESOLVED="claude,opencode,github-copilot"
    OPENSPEC_INIT_ARGS=(--tools "$OPENSPEC_TOOLS_RESOLVED")
    [ -n "$OPENSPEC_PROFILE" ] && OPENSPEC_INIT_ARGS+=(--profile "$OPENSPEC_PROFILE")
    say "  openspec init --tools $OPENSPEC_TOOLS_RESOLVED${OPENSPEC_PROFILE:+ --profile $OPENSPEC_PROFILE}…"
    ( cd "$TARGET" && openspec init "${OPENSPEC_INIT_ARGS[@]}" </dev/null ) \
      && say "  ✓ openspec init ($OPENSPEC_TOOLS_RESOLVED${OPENSPEC_PROFILE:+, profile $OPENSPEC_PROFILE})" \
      || say "  ⚠ openspec init best-effort — manual: openspec init --tools $OPENSPEC_TOOLS_RESOLVED"
  fi
}

# ── Install the chosen engine ──────────────────────────────────────────────────
say ""; say "[1/1] SDD engine: $ENGINE"
case "$ENGINE" in
  gsd)       install_gsd ;;
  spec-kit)  install_speckit ;;
  openspec)  install_openspec ;;
esac

# ── Readiness ──────────────────────────────────────────────────────────────────
say ""
say "════════════════════════════════════════════════════════"
say " SDD engine readiness — $ENGINE — $(date '+%Y-%m-%d')"
say "════════════════════════════════════════════════════════"
case "$ENGINE" in
  gsd)
    for rt in ${GSD_TARGETS:-claude}; do
      d="$(gsd_dir "$rt")"; [ -n "$d" ] || continue
      printf "  %-24s %s\n" "GSD Core ($rt):" "$(cat "$d/gsd-core/VERSION" 2>/dev/null || echo pending) · $(ls "$d/skills" 2>/dev/null | grep -c '^gsd-') skills in $d/skills"
    done
    printf "  %-24s %s\n" "pinned:" "${GSD_VERSION:-sdd.gsd.version missing}"
    printf "  %-24s %s\n" "node:" "$(node --version 2>/dev/null || echo missing) (GSD Core needs >= 24)"
    # gsd-ship reads the ROADMAP record with jq and opens the PR with gh: without them nothing ships.
    printf "  %-24s %s\n" "jq:" "$(have jq && echo present || echo 'missing — gsd-ship needs it; install it with your package manager')"
    ;;
  spec-kit)
    printf "  %-24s %s\n" "specify:" "$(have specify && echo installed || echo pending)"
    printf "  %-24s %s\n" "agents:"  "${SPECKIT_AGENTS:-claude opencode copilot}"
    ;;
  openspec)
    printf "  %-24s %s\n" "openspec:" "$(have openspec && echo "installed ($(openspec --version 2>/dev/null | head -1))" || echo pending)"
    ;;
esac
say ""
say "Done. The adapter playbook is $TARGET/.factory/sdd/$ENGINE/ADAPTER.md — /factory-discover and /factory-build read it."
[ "$PENDING" = 1 ] && { say "Waiting for the user's confirmation (see the → line above)."; exit 3; }
exit 0
