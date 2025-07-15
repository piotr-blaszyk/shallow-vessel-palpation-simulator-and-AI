import json

class ScientificNotationEncoder(json.JSONEncoder):
    def default(self, obj):
        return super().default(obj)
    
    def iterencode(self, obj, _one_shot=False):
        for s in super().iterencode(obj, _one_shot=False):
            try:
                float(s)
                if '.' in s or 'e' in s.lower():
                    value = float(s)
                    s = f"{value:.3e}"
            except (ValueError, TypeError):
                pass
            yield s

def main():
    source = "difftactile/system_params/system-params-distances.json"
    target_raw = "difftactile/system_params/system-params.json"
    target_computed = "difftactile/system_params/system-params.json"
    with open(source, 'r') as f:
        source_data = json.load(f)
    with open(target_raw, 'r') as f:
        target_data = json.load(f)
    scaling_factor = target_data['meta']['distance_scaling_factor']
    def update_distances(source_dict, target_dict):
        for key, value in source_dict.items():
            if isinstance(value, dict):
                if key not in target_dict:
                    target_dict[key] = {}
                update_distances(value, target_dict[key])
            else:
                if isinstance(value, (int, float)):
                    target_dict[key] = value * scaling_factor
    update_distances(source_data, target_data)
    with open(target_computed, 'w') as f:
        json.dump(target_data, f, indent=4, cls=ScientificNotationEncoder)

if __name__ == "__main__":
    main()
