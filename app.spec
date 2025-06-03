# -*- mode: python ; coding: utf-8 -*-
import os
import whisper
import site

block_cipher = None

# Find the path to the whisper model assets
whisper_path = os.path.dirname(whisper.__file__)
whisper_assets = os.path.join(whisper_path, 'assets')

a = Analysis(
    ['app_wrapper.py'],  # Main script to execute
    pathex=[],
    binaries=[],
    datas=[
        ('resemblyzer', 'resemblyzer'),
                (whisper_assets, 'whisper/assets'),  # Include whisper assets

        ('static', 'static'),        # Include static directory

    ],
    hiddenimports=[
        'app',
        'app_wrapper',
        'auth_routes',
        'auth_service',
        'config',
        'db_models',
        'presentation_routes',
        'presentation_service',
        'profile_routes',
        'run',
        'template_manager',
        'utils',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(
    a.pure, 
    a.zipped_data,
    cipher=block_cipher
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Memories AI',
)