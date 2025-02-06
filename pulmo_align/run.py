#!/usr/bin/env python3
"""
Script ejecutable para PulmoAlign Viewer.
"""

import sys
import os
from pathlib import Path
from pulmo_align.viewer import main

if __name__ == "__main__":
    # Configurar display para X11
    if 'DISPLAY' not in os.environ or os.environ['DISPLAY'] == '10.255.255.254:0':
        os.environ['DISPLAY'] = ':0'
    
    main()
