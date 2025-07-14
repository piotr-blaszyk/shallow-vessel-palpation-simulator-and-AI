import json

def main():
    source = "difftactile/system_params/system-params-distances.json"
    target = "difftactile/system_params/system-params.json"
    with open(source, 'r') as f:
        source_data = json.load(f)
    with open(target, 'r') as f:
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
    with open(target, 'w') as f:
        json.dump(target_data, f, indent=4)

if __name__ == "__main__":
    main()
