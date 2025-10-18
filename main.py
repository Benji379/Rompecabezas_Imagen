# main.py
import os, io, copy, re, json
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import cairosvg
from flask import Flask, request, send_from_directory, jsonify, send_file

OUT_DIR = "piezas_out"
UPLOAD_DIR = "uploads"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

PIECES_EXPECTED = 12
AREA_MIN_RATIO = 0.002
AREA_BG_RATIO  = 0.90
TYPES = {"path","rect","polygon","ellipse","circle","polyline"}

def get_svg_dimensions(root):
    w_attr = root.attrib.get("width", "")
    h_attr = root.attrib.get("height", "")
    viewBox = root.attrib.get("viewBox", "")
    def _p(v):
        if not v or v.endswith("%"): return None
        m = re.match(r"^\s*([\d\.]+)", v)
        return int(float(m.group(1))) if m else None
    w = _p(w_attr); h = _p(h_attr)
    if (w is None or h is None) and viewBox:
        vb = [float(x) for x in viewBox.strip().split()]
        if len(vb)==4:
            w = w or int(round(vb[2])); h = h or int(round(vb[3]))
    return (w or 1024, h or 768)

def deep_svg_with_single_element(base_root, element):
    new_root = ET.Element(base_root.tag, base_root.attrib)
    W, H = get_svg_dimensions(base_root)
    new_root.set("width", str(W)); new_root.set("height", str(H))
    el_copy = copy.deepcopy(element)
    el_copy.attrib.pop("stroke", None)
    el_copy.attrib["fill"] = "#ffffff"
    if "style" in el_copy.attrib:
        style = el_copy.attrib["style"]
        style = re.sub(r"stroke\s*:\s*[^;]+", "stroke:none", style, flags=re.I)
        style = re.sub(r"fill\s*:\s*[^;]+", "fill:#ffffff", style, flags=re.I)
        el_copy.set("style", style)
    new_root.append(el_copy)
    return new_root

def render_mask_from_root(svg_root):
    W, H = get_svg_dimensions(svg_root)
    bytes_svg = ET.tostring(svg_root, encoding="utf-8", method="xml")
    png_bytes = cairosvg.svg2png(bytestring=bytes_svg, output_width=W, output_height=H)
    img = Image.open(io.BytesIO(png_bytes)).convert("L")
    arr = np.array(img)
    binm = (arr > 10).astype(np.uint8) * 255
    ys, xs = np.where(binm>0)
    if ys.size == 0:
        return binm, None, 0, (W,H)
    bbox = (xs.min(), ys.min(), xs.max()+1, ys.max()+1)
    area = int((binm>0).sum())
    return binm, bbox, area, (W,H)

def build_pieces(PNG_PATH, SVG_PATH, out_dir=OUT_DIR):
    tree = ET.parse(SVG_PATH)
    root = tree.getroot()
    W, H = get_svg_dimensions(root)
    root.set("width", str(W)); root.set("height", str(H))
    canvas_area = W * H

    # Candidatas
    candidates = []
    for el in root.iter():
        tag = el.tag.split('}')[-1].lower()
        if tag in TYPES and el.attrib.get("display","") != "none":
            candidates.append(el)

    # Medir
    scored = []
    for idx, el in enumerate(candidates):
        single = deep_svg_with_single_element(root, el)
        mask_np, bbox, area, _sz = render_mask_from_root(single)
        if bbox is None: continue
        ratio = area / canvas_area
        if ratio >= AREA_BG_RATIO or ratio < AREA_MIN_RATIO: continue
        scored.append((area, bbox, mask_np, idx))

    if not scored:
        raise RuntimeError("No se detectaron figuras útiles en el SVG.")
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:PIECES_EXPECTED]

    # Unión de bboxes
    ux0, uy0, ux1, uy1 = W, H, 0, 0
    for _, bbox, _, _ in top:
        x0, y0, x1, y1 = bbox
        ux0 = min(ux0, x0); uy0 = min(uy0, y0)
        ux1 = max(ux1, x1); uy1 = max(uy1, y1)
    PAD = 1
    ux0 = max(0, ux0 - PAD); uy0 = max(0, uy0 - PAD)
    ux1 = min(W, ux1 + PAD); uy1 = min(H, uy1 + PAD)
    uW, uH = max(1, ux1 - ux0), max(1, uy1 - uy0)

    # PNG: contain dentro del área de unión
    src_in = Image.open(PNG_PATH).convert("RGBA")
    scale = min(uW / src_in.width, uH / src_in.height)
    new_w = max(1, int(round(src_in.width * scale)))
    new_h = max(1, int(round(src_in.height * scale)))
    src_scaled = src_in.resize((new_w, new_h), Image.LANCZOS)
    src = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    off_x = ux0 + (uW - new_w) // 2
    off_y = uy0 + (uH - new_h) // 2
    src.paste(src_scaled, (off_x, off_y))

    # Limpiar outdir
    for f in os.listdir(out_dir):
      try: os.remove(os.path.join(out_dir, f))
      except: pass

    metadata = { "canvas": {"width": int(W), "height": int(H)}, "pieces": [] }
    for i, (area, bbox, mask_np, idx) in enumerate(top, start=1):
        x0, y0, x1, y1 = [int(v) for v in bbox]
        w, h = int(x1 - x0), int(y1 - y0)
        mask_img = Image.fromarray(mask_np, mode="L")
        piece_full = Image.new("RGBA", (W, H))
        piece_full.paste(src, mask=mask_img)
        piece_crop = piece_full.crop((x0, y0, x1, y1))
        file_name = f"pieza_{i}.png"
        piece_crop.save(os.path.join(out_dir, file_name))
        metadata["pieces"].append({
            "id": i, "file": file_name, "x": x0, "y": y0, "w": w, "h": h,
            "nx": x0 / W, "ny": y0 / H, "nw": w / W, "nh": h / H
        })

    with open(os.path.join(out_dir, "pieces_map.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return {"exported": len(top), "map": "piezas_out/pieces_map.json"}

app = Flask(__name__, static_folder=".", static_url_path="")

@app.route("/")
def root_index():
    return send_file("index.html")

@app.route("/piezas_out/<path:fname>")
def serve_pieces(fname):
    return send_from_directory(OUT_DIR, fname)

@app.route("/api/build", methods=["POST"])
def api_build():
    if "svg" not in request.files or "png" not in request.files:
        return jsonify({"ok": False, "error": "Envía campos 'svg' y 'png'"}), 400
    svg_path = os.path.join(UPLOAD_DIR, "input.svg")
    png_path = os.path.join(UPLOAD_DIR, "input.png")
    request.files["svg"].save(svg_path)
    request.files["png"].save(png_path)
    try:
        result = build_pieces(png_path, svg_path, OUT_DIR)
        result["map_url"] = f"/{result['map']}?t={int(np.random.randint(1e9))}"
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
