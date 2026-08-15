"""Bird's-eye vessel map for one configuration. Prefer ./docker/vessel_map.sh,
which sets the DIFFTACTILE_MAP_* environment variables this reads."""
from difftactile.data_analysis.experiment.vessel_map import main

if __name__ == "__main__":
    main()
