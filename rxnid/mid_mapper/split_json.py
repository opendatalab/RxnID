import json, os, argparse, math

def split_json(input_path, output_dir, num_splits):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    images = data.get('images', [])
    total = len(images)
    if total == 0:
        print("[WARN] images 为空，跳过拆分")
        return []
    os.makedirs(output_dir, exist_ok=True)
    part_paths = []
    per = math.ceil(total / num_splits)
    for i in range(num_splits):
        start = i * per
        end = min(total, (i + 1) * per)
        if start >= end:
            break
        part = {
            "licenses": data.get("licenses", []),
            "info": data.get("info", {}),
            "categories": data.get("categories", []),
            "images": images[start:end]
        }
        out_path = os.path.join(output_dir, f"part_{i}.json")
        with open(out_path, 'w', encoding='utf-8') as wf:
            json.dump(part, wf, ensure_ascii=False, indent=2)
        part_paths.append(out_path)
        print(f"[INFO] 写出 {out_path} (images {start}~{end-1}, 共 {end-start})")
    return part_paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--num_splits', type=int, default=4)
    args = ap.parse_args()
    split_json(args.input, args.output_dir, args.num_splits)

if __name__ == '__main__':
    main()
