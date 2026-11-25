#!/usr/bin/env python3
"""Make the ChArUco calibration board: printable PNG at exact scale + an Isaac asset that shows it.

ChArUco = a chessboard whose white squares carry ArUco markers. The markers make every corner
uniquely identifiable even when part of the board is out of view; the chessboard corners give
sub-pixel precision. Real cell: print the PNG at 100 % on a rigid flat plate and measure one
square with calipers; put the measured size into cell.yaml.

  python3 ~/ip_ws/src/tools/make_charuco.py      # writes ~/ip_ws/assets/calib/charuco.png and charuco_board.usda
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "lib"))

import os, cv2, numpy as np, yaml

CFG = os.path.expanduser("~/ip_ws/config/cell.yaml")
OUT = os.path.expanduser("~/ip_ws/assets/calib")
BOARD = {"squares": [7, 5], "square": 0.04, "marker": 0.03, "dictionary": "DICT_5X5_100", "px_per_m": 10000}


from charuco import board_from


def main():
    os.makedirs(OUT, exist_ok=True)
    b, _ = board_from(BOARD)
    w, h = (int(round(BOARD["squares"][i] * BOARD["square"] * BOARD["px_per_m"])) for i in (0, 1))
    img = b.draw((w, h), marginSize=0, borderBits=1)
    cv2.imwrite(f"{OUT}/charuco.png", img)
    W, H = BOARD["squares"][0] * BOARD["square"], BOARD["squares"][1] * BOARD["square"]
    # A flat quad with the image mapped un-mirrored when seen from above (+Z), plus a dark backing plate.
    usda = f'''#usda 1.0
(
    defaultPrim = "CharucoBoard"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "CharucoBoard" (
    doc = "ChArUco {BOARD['squares'][0]}x{BOARD['squares'][1]}, square {BOARD['square']} m, marker {BOARD['marker']} m, {BOARD['dictionary']}. Board face at z = +0.0005, origin at the board centre."
)
{{
    def Mesh "Face"
    {{
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [({-W/2}, {-H/2}, 0.0005), ({W/2}, {-H/2}, 0.0005), ({W/2}, {H/2}, 0.0005), ({-W/2}, {H/2}, 0.0005)]
        normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = "vertex")
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (interpolation = "vertex")
        uniform token subdivisionScheme = "none"
        rel material:binding = </CharucoBoard/Looks/BoardMat>
    }}
    def Cube "Backing"
    {{
        double size = 1
        float3 xformOp:scale = ({W + 0.02}, {H + 0.02}, 0.004)
        double3 xformOp:translate = (0, 0, -0.0015)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [(0.05, 0.05, 0.05)]
    }}
    def Scope "Looks"
    {{
        def Material "BoardMat"
        {{
            token outputs:surface.connect = </CharucoBoard/Looks/BoardMat/Surface.outputs:surface>
            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </CharucoBoard/Looks/BoardMat/Tex.outputs:rgb>
                float inputs:roughness = 0.7
                float inputs:metallic = 0
                token outputs:surface
            }}
            def Shader "Tex"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @./charuco.png@
                float2 inputs:st.connect = </CharucoBoard/Looks/BoardMat/St.outputs:result>
                token inputs:wrapS = "clamp"
                token inputs:wrapT = "clamp"
                float3 outputs:rgb
            }}
            def Shader "St"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }}
        }}
    }}
}}
'''
    open(f"{OUT}/charuco_board.usda", "w").write(usda)
    # record the board in the config
    cfg = yaml.safe_load(open(CFG))
    cfg_txt = open(CFG).read()
    if "calibration:" not in cfg_txt:
        cfg_txt += ("\ncalibration:\n  charuco:                 # the board used for hand-eye calibration. Real cell: measure the printed square with calipers\n"
                    f"    squares: [{BOARD['squares'][0]}, {BOARD['squares'][1]}]\n    square: {BOARD['square']}           # m, chessboard square side\n"
                    f"    marker: {BOARD['marker']}           # m, ArUco marker side\n    dictionary: {BOARD['dictionary']}\n")
        open(CFG, "w").write(cfg_txt)
    print(f"board {BOARD['squares'][0]} x {BOARD['squares'][1]} squares, {W*1000:.0f} x {H*1000:.0f} mm, image {w} x {h} px")
    print(f"wrote {OUT}/charuco.png and {OUT}/charuco_board.usda; board recorded in cell.yaml")


if __name__ == "__main__":
    main()
