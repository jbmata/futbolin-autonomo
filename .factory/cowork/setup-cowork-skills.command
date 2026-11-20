#!/usr/bin/env bash
# setup-cowork-skills.command - lanzador para macOS (doble clic en Finder).
# macOS ejecuta los ficheros .command en Terminal al hacer doble clic. Este wrapper
# solo delega en setup-cowork-skills.sh: macOS es un entorno Unix/bash, no necesita un
# script aparte (a diferencia de Windows, que usa setup-cowork-skills.ps1).
#
# Tambien puedes ejecutarlo desde Terminal:  bash setup-cowork-skills.sh
# Si falta npx/Node:  brew install node
cd "$(dirname "$0")" || exit 1
exec bash ./setup-cowork-skills.sh
