"""Validate the single-sheet STM32H733 / ATX / eight-TMC2209 controller."""
import argparse
import csv
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kicad-cli',default=shutil.which('kicad-cli') or r'C:\Program Files\KiCad\10.0\bin\kicad-cli.exe')
    parser.add_argument('--cube-db',type=Path,default=Path(r'C:\Users\info\AppData\Local\Programs\STM32CubeMX\db\mcu'))
    args=parser.parse_args()
    project=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='mcu-rt-check-') as temporary:
        work=Path(temporary)
        env=dict(os.environ,KICAD_CONFIG_HOME=str(work/'config'),KICAD_DOCUMENTS_HOME=str(work))
        for command in [['sch','export','netlist','--format','kicadxml','-o',str(work/'design.xml')],['sch','erc','-o',str(work/'erc.rpt')]]:
            result=subprocess.run([args.kicad_cli,*command,str(project/'MCU-RT.kicad_sch')],env=env,capture_output=True,text=True,timeout=180)
            assert result.returncode==0,result.stderr
        report=(work/'erc.rpt').read_text(encoding='utf-8')
        assert re.search(r'Errors 0\s+Warnings 10',report),report
        assert re.findall(r'^\[([^]]+)\]',report,re.M)==['isolated_pin_label']*10,report
        assert len(re.findall(r"Label 'ETH_[^']+'",report))==10,report
        design=ET.parse(work/'design.xml').getroot()

    components={c.get('ref'):c for c in design.findall('components/comp')}
    assert len(design.findall('design/sheet'))==1
    assert {c.find('sheetpath').get('names') for c in components.values()}=={'/'}
    assert {ref for ref in components if ref.startswith('A')}=={f'A{i*100+1}' for i in range(1,9)}
    assert components['U5'].find('libsource').get('part')=='STM32H733ZGTx'
    assert components['U5'].findtext('value')=='STM32H733ZGT6'
    assert components['U5'].findtext('footprint')=='Package_QFP:LQFP-144_20x20mm_P0.5mm'
    assert not any('NUCLEO' in c.find('libsource').get('part') for c in components.values())
    assert not any(re.fullmatch(r'F\d+',ref) for ref in components)
    schematic=(project/'MCU-RT.kicad_sch').read_text(encoding='utf-8')
    assert '(global_label' not in schematic and '(paper "A1")' in schematic
    nets={(n.get('ref'),n.get('pin')):net.get('name').lstrip('/') for net in design.findall('nets/net') for n in net.findall('node')}
    def net(ref,pin):
        return nets[(ref,str(pin))]
    def same(*pins):
        names=[net(*p) for p in pins]
        assert len(set(names))==1,(pins,names)
    def separated(*pins):
        names=[net(*p) for p in pins]
        assert len(set(names))==len(names),(pins,names)
    def unused(ref,pin): assert net(ref,pin).startswith('unconnected-'),(ref,pin,net(ref,pin))

    for pin in [4,6,21,22,23]: same(('J1',pin),('J1',4),('C2',1))
    for pin in [1,2,12,13]: same(('J1',pin),('TP3',1))
    assert {pin for pin,name in nets.items() if name==net('J1',1)}=={('J1',str(pin)) for pin in [1,2,12,13]}|{('TP3','1')}
    for pin in [3,5,7,15,17,18,19,24]: same(('J1',pin),('J1',3))
    same(('J1',8),('TP1',1));same(('J1',9),('TP2',1))
    same(('J1',10),('J1',11),('TP4',1))
    same(('J1',16),('J3',1));same(('J3',2),('J1',3))
    unused('J1',14);unused('J1',20)
    for pin in range(1,5): same(('J2',pin),('J1',3))
    for pin in range(5,9): same(('J2',pin),('C1',1),('D1',1))
    same(('D1',2),('J1',3))
    for ref in ['J4','J5']:
        same((ref,1),('J1',3));same((ref,2),('J2',5))
        assert {pin for r,pin in nets if r==ref}=={'1','2'}
        assert components[ref].findtext('footprint')=='Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical'
    assert components['D1'].find('libsource').get('part')=='D_Zener'
    separated(('J1',1),('U10',1),('J1',9),('J1',4),('J1',10),('J2',5))
    same(('U5',59),('R1',1));same(('U5',58),('R1',2))
    same(('U5',119),('R2',1));same(('U5',122),('R2',2))
    separated(('R1',1),('R1',2),('R2',1),('R2',2))

    rows=list(csv.DictReader((project/'Pinout.csv').open(encoding='utf-8')))
    assert len(rows)==65 and len({r['GPIO'] for r in rows})==65
    signals={r['Signal']:r for r in rows}
    db=ET.parse(args.cube_db/'STM32H733ZGTx.xml').getroot()
    ns={'s':db.tag.split('}')[0][1:]}
    pins={re.match(r'P[A-H]\d+',p.get('Name'))[0]:p for p in db.findall('s:Pin',ns) if re.match(r'P[A-H]\d+',p.get('Name'))}
    gpio_ip=next(p.get('Version') for p in db.findall('s:IP',ns) if p.get('Name')=='GPIO')
    af=ET.parse(args.cube_db/'IP'/f'GPIO-{gpio_ip}_Modes.xml').getroot()
    afpins={re.match(r'P[A-H]\d+',p.get('Name'))[0]:p for p in af.findall('s:GPIO_Pin',ns) if re.match(r'P[A-H]\d+',p.get('Name'))}
    for r in rows:
        assert pins[r['GPIO']].get('Position')==r['LQFP144'],r
        assert r['Function'] in [s.get('Name') for s in pins[r['GPIO']].findall('s:Signal',ns)],r
        if r['Signal'] not in ('SWDIO','SWCLK'): assert net('U5',r['LQFP144'])==r['Net'],r
        if r['AlternateFunction']:
            signal=afpins[r['GPIO']].find(f's:PinSignal[@Name="{r["Function"]}"]/s:SpecificParameter[@Name="GPIO_AF"]/s:PossibleValue',ns)
            assert signal.text==r['AlternateFunction'],r
    assert len({r['Function'].split('_')[0] for r in rows if r['Signal'].endswith('_STEP')})==8
    for i in range(1,9):
        b=i*100;a=f'A{b+1}';address=(i-1)%4
        for suffix,pin in [('STEP',2),('DIR',1),('EN_N',8)]:
            r=signals[f'MOTOR_{i}_{suffix}'];same((a,pin),('U5',r['LQFP144']))
        same((a,8),(f'R{b+1}',2));same((f'R{b+1}',1),('U10',1))
        same((a,2),(f'R{b+2}',1));same((a,1),(f'R{b+3}',1))
        same((a,3),(a,10),(a,16),(f'R{b+2}',2),(f'R{b+3}',2),('J1',3))
        same((a,15),('U10',1),(f'C{b+3}',1))
        same((a,5),('R1' if i<=4 else 'R2',2));unused(a,4)
        for bit,pin in [(0,7),(1,6)]:
            same((a,pin),(f'R{b+6+bit}',2))
            same((f'R{b+6+bit}',1),('U10',1) if address&(1<<bit) else ('J1',3))
        separated((a,6),(a,7),(a,1),(a,2),(a,8))
        same(('J2',5),(a,9),(f'C{b+1}',1),(f'C{b+2}',1))
        assert components[f'C{b+1}'].findtext('footprint').startswith('Capacitor_THT:')
        for terminal,pin in enumerate([11,12,13,14],1):same((a,pin),(f'J{b+1}',terminal))
        assert components[f'J{b+1}'].findtext('footprint')=='Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical'
    same(*[(f'A{i*100+1}',9) for i in range(1,9)])
    limits=[signals[f'LIMIT_{i}'] for i in range(1,9)]
    assert [r['GPIO'] for r in limits]==['PD0','PD1','PD2','PD3','PD7','PF8','PF9','PD10']
    assert len({int(r['GPIO'][2:]) for r in limits})==8
    assert not {'U1','U2','C3','C4',*(f'D{900+i}' for i in range(1,9))} & components.keys()
    for i,r in enumerate(limits,1):
        j=f'J{900+i}';rp=f'R{899+2*i}';rs=f'R{900+2*i}';c=f'C{900+i}'
        same((j,1),('J1',3),(c,2))
        same((j,2),(rp,2),(rs,1));same((rp,1),('U10',1))
        same((rs,2),(c,1),('U5',r['LQFP144']))
        separated((j,2),(rs,2),('U10',1),('JP1',2),('J1',3))
        assert components[j].findtext('footprint')=='Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical'
        assert components[rp].findtext('value')=='2.2k' and components[rs].findtext('value')=='1k'
        assert components[c].findtext('value')=='100n 16V'
    # Eight identical fixed-address external encoders need isolated mux branches.
    assert components['U3'].findtext('value')=='TCA9548APWR'
    assert components['U3'].findtext('footprint')=='Package_SO:TSSOP-24_4.4x7.8mm_P0.65mm'
    mux=next(c for c in design.findall('libparts/libpart') if c.get('part')=='TCA9548APWR')
    assert {pin.get('num'):pin.get('name') for pin in mux.findall('pins/pin')}=={
        '1':'A0','2':'A1','3':'~{RESET}','4':'SD0','5':'SC0','6':'SD1','7':'SC1',
        '8':'SD2','9':'SC2','10':'SD3','11':'SC3','12':'GND','13':'SD4','14':'SC4',
        '15':'SD5','16':'SC5','17':'SD6','18':'SC6','19':'SD7','20':'SC7',
        '21':'A2','22':'SCL','23':'SDA','24':'VCC'}
    same(('U3',1),('U3',2),('U3',21),('U3',12),('J1',3))
    same(('U3',24),('C1101',1),('C1102',1),('R1117',1),('R1118',1),('R1119',1),('U10',1))
    same(('C1101',2),('C1102',2),('J1',3))
    assert components['C1101'].findtext('value')=='100n 16V'
    assert components['C1102'].findtext('value')=='1uF'
    same(('U3',22),('U5',136),('R1117',2))
    same(('U3',23),('U5',137),('R1118',2))
    same(('U3',3),('U5',118),('R1119',2))
    for r in ['R1117','R1118']:assert components[r].findtext('value')=='2.2k'
    assert components['R1119'].findtext('value')=='10k'
    assert signals['ENC_MUX_RESET_N']['GPIO']=='PD4'
    for signal,gpio,pin,fn in [('ENC_I2C_SCL','PB6','136','I2C1_SCL'),('ENC_I2C_SDA','PB7','137','I2C1_SDA')]:
        row=signals[signal];assert (row['GPIO'],row['LQFP144'],row['Function'],row['AlternateFunction'])==(gpio,pin,fn,'GPIO_AF4_I2C1')
    branches=[(5,4),(7,6),(9,8),(11,10),(14,13),(16,15),(18,17),(20,19)]
    separated(*[('U3',pin) for pair in branches for pin in pair],('U3',22),('U3',23),('U3',3),('U10',1),('J1',3))
    for i,(sc,sd) in enumerate(branches,1):
        j=f'J{1100+i}';rsda=f'R{1099+2*i}';rscl=f'R{1100+2*i}'
        same((j,1),('J1',3));same((j,2),(rsda,1),(rscl,1),('U10',1))
        same((j,3),('U3',sd),(rsda,2));same((j,4),('U3',sc),(rscl,2))
        for r in [rsda,rscl]:assert components[r].findtext('value')=='4.7k'
        assert components[j].findtext('footprint')=='Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical'
    # With one channel selected, allow for the pictured module's 10k pull-ups.
    assert 3.3*(1/2200+1/4700+1/10000)<.003

    # Bare MCU supply, reset, clock and programming connections.
    same(('U10',1),('U10',2),('C1002',1))
    same(('U10',6),('U10',5),('JP1',2),('C1001',1))
    same(('U10',3),('U10',7),('J1',3))
    unused('U10',4)
    for pin in [6,17,30,33,39,52,62,72,84,95,108,121,131,143,144]:same(('U5',pin),('U10',1))
    for pin in [16,31,38,51,61,83,94,107,120,130]:same(('U5',pin),('J1',3))
    for pin,c in [(71,'C1029'),(106,'C1030')]:
        same(('U5',pin),(c,1));same((c,2),('J1',3))
        separated(('U5',pin),('U10',1),('J1',3))
        assert components[c].findtext('value')=='2.2uF'
    same(('U5',138),('R1004',1),('SW1',1));same(('R1004',2),('J1',3));same(('SW1',2),('U10',1))
    same(('U5',23),('Y10',1),('C1005',1));same(('U5',24),('Y10',3),('C1006',1))
    same(('Y10',2),('Y10',4),('C1005',2),('C1006',2),('J1',3))
    separated(('U5',23),('U5',24),('J1',3))
    assert components['Y10'].findtext('value')=='24MHz'
    same(('U5',32),('R1007',2),('C1019',2),('C1020',2));same(('R1007',1),('U10',1))
    for header,mcu,r in [(3,105,'R1006'),(4,109,'R1008')]:
        same(('J7',header),(r,2));same((r,1),('U5',mcu));assert components[r].findtext('value')=='33R'
    same(('J7',1),('U10',1));same(('J7',2),('J1',3));same(('J7',6),('U5',133))
    same(('J7',5),('R1003',1),('R1040',2),('SW2',2))
    same(('R1003',2),('U5',25));same(('R1040',1),('U10',1));same(('SW2',1),('J1',3))
    assert 'C1033' not in components  # User removed the reset capacitor; still pending restoration.
    assert components['J7'].findtext('footprint')=='Connector_PinHeader_2.54mm:PinHeader_1x06_P2.54mm_Vertical'
    same(('J8',1),('J1',3));same(('J8',2),('U5',77));same(('J8',3),('U5',78));same(('J8',4),('U10',1))
    assert components['J8'].findtext('footprint')=='Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical'
    assert components['J10'].findtext('footprint')=='Connector_PinHeader_2.54mm:PinHeader_2x20_P2.54mm_Vertical'
    for pin in range(1,41):
        if pin in [6,9,14,20,25,30,34,39]:same(('J10',pin),('J1',3))
        elif pin in [2,4]:same(('J10',pin),('J1',4),('JP1',1))
        elif pin in [8,10]:same(('J10',pin),('U5',141 if pin==8 else 142))
        elif pin in [19,21,23,24]:same(('J10',pin),('U5',{19:67,21:66,23:65,24:64}[pin]))
        else:unused('J10',pin)
    for signal,gpio,pin in [('MOSI','PE14',67),('MISO','PE13',66),('SCK','PE12',65),('NSS','PE11',64)]:
        row=signals['RPI_SPI_'+signal]
        assert (row['GPIO'],row['LQFP144'],row['Function'],row['AlternateFunction'])==(gpio,str(pin),'SPI4_'+signal,'GPIO_AF5_SPI4'),row
    separated(*[('J10',pin) for pin in [19,21,23,24,8,10]],('U10',1),('J1',1),('J1',3))
    separated(('J10',2),('JP1',2),('JP1',3),('U10',1),('J1',9))
    separated(('J10',8),('J10',10),('J8',2),('J8',3),('R1',2),('R2',2))

    # User USB-C circuit, internal FS PHY, and mutually exclusive 5V source selector.
    same(('U5',103),('R32',1));same(('R32',2),('J17','A7'),('J17','B7'))
    same(('U5',104),('R33',1));same(('R33',2),('J17','A6'),('J17','B6'))
    for r in ['R32','R33']:assert components[r].findtext('value')=='0R'
    same(('U5',101),('R34',2),('R35',1))
    assert signals['USB_VBUS_SENSE']['Function']=='GPIO'
    same(('J17','A5'),('R4',1));same(('J17','B5'),('R3',1))
    separated(('J17','A5'),('J17','B5'))
    for r in ['R3','R4']:
        assert components[r].findtext('value')=='5k1';same((r,2),('J1',3))
    assert components['R34'].findtext('value')=='33k 1%' and components['R35'].findtext('value')=='82k 1%'
    same(('J17','A4'),('J17','A9'),('J17','B4'),('J17','B9'),('R34',1),('D2',2))
    same(('J17','SH'),('J17','A1'),('J17','B1'),('J17','A12'),('J17','B12'),('R35',2),('J1',3))
    unused('J17','A8');unused('J17','B8')
    same(('JP1',1),('J1',4),('C2',1));same(('JP1',3),('D2',1))
    separated(('JP1',1),('JP1',2),('JP1',3),('D2',2),('U10',1),('J2',5),('J1',3))
    assert components['JP1'].findtext('footprint')=='Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical'
    assert components['D2'].findtext('footprint')=='MCU-RT:LSM115J_SMB'
    diode=next(c for c in design.findall('libparts/libpart') if c.get('part')=='LSM115JE3_TR13')
    assert {pin.get('num'):pin.get('name') for pin in diode.findall('pins/pin')}=={'1':'K','2':'A'}
    # A single cap joins only one pair; the other supply remains isolated in either position.
    for pair,source,other in [(('1','2'),('J1',4),('D2',1)),(('2','3'),('D2',1),('J1',4))]:
        joined={net('JP1',pin) for pin in pair}
        assert net('U10',6) in joined and net(*source) in joined and net(*other) not in joined
    same(('C1031',1),('C1032',1),('U5',95));same(('C1031',2),('C1032',2),('J1',3))
    assert not {'J6','U11','U12','U13','J9','Y11'} & components.keys()

    kicad=Path(args.kicad_cli).resolve().parents[1]/'share/kicad/footprints'
    for ref,c in components.items():
        if ref.startswith('R'): assert c.findtext('footprint').startswith('Resistor_SMD:R_0805_2012Metric'),ref
        if ref.startswith('C') and ref not in {'C1','C2211',*(f'C{i*100+1}' for i in range(1,9))}:
            assert c.findtext('footprint')=='Capacitor_SMD:C_0805_2012Metric',ref
        library,name=c.findtext('footprint').split(':',1)
        path=(project/'Libraries/MCU-RT.pretty' if library=='MCU-RT' else kicad/(library+'.pretty'))/(name+'.kicad_mod')
        content=path.read_text(encoding='utf-8')
        pads={a or b for a,b in re.findall(r'\(pad\s+(?:"([^"]*)"|([^\s()]+))\s',content)}-{''}
        assert {pin for r,pin in nets if r==ref}<=pads,(ref,path)
        if ref in ('J1','J2'):
            positions={int(p):(float(x),float(y)) for p,x,y in re.findall(r'\(pad "(\d+)"[^\n]*\s+\(at ([\d.]+) ([\d.]+)\)',content)}
            count=12 if ref=='J1' else 4
            assert all(positions[i][1]==0 for i in range(1,count+1))
            assert all(positions[i][1]==5.5 for i in range(count+1,2*count+1))
    print(f'PASS: one A1 sheet; local labels; {len(components)} components; H733 power/reset/boot (reset capacitor absent), USB/source selector, clock, 8 motors, 8 direct endstops, 8 AS5600 ports/TCA9548A, ST-LINK, USART3 and Raspberry Pi UART8/SPI4; 65 GPIO/AF checked (10 Ethernet reservations, PHY removed).')
    print(re.search(r'ERC messages:.*',report)[0])


if __name__=='__main__':
    main()
