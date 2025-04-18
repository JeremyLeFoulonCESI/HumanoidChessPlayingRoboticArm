# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['src\\hcpra.py', 'src\\application.py', 'src\\board_display.py', 'src\\changelist.py', 'src\\chess_game.py', 'src\\color_board.py', 'src\\debug_info.py', 'src\\errors.py', 'src\\image.py', 'src\\image_analysis.py', 'src\\mutable_fraction.py', 'src\\robot.py', 'src\\stockfish_implementation.py', 'src\\window.py'],
    pathex=[],
    binaries=[],
    datas=[('src/stockfish-windows-x86-64-avx2.exe', '.'), ('src/camera_calibration.json', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='hcpra',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    hide_console='minimize-late',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='hcpra',
)
