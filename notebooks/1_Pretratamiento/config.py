import yaml
import pandas as pd
from pathlib import Path

class Config:
    def __init__(self, config_file="../config.yml"):
        # Load configuration from YAML file
        with open(config_file, "r", encoding='utf8') as ymlfile:
            self.cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        
        # Rutas de entrada y salida de datos
        self.path_in = Path(self.cfg['rutas']['anuario_aforos']['orig'])
        self.path_out = Path(self.cfg['rutas']['anuario_aforos'].get('repo', '../../data/anuario_aforos'))
        self.path_GIS = self.path_out / 'GIS'
        self.path_plots = self.path_out / 'plots'

        # Ensure directories exist
        for path in [self.path_out, self.path_GIS, self.path_plots]:
            path.mkdir(parents=True, exist_ok=True)

        # Inicio y fin del periodo de estudio
        cfg_camels = self.cfg['CAMELS-ES']
        self.start = pd.to_datetime(cfg_camels['periodo'].get('inicio', None))
        self.end = pd.to_datetime(cfg_camels['periodo'].get('final', None))

        # Tamaño mínimo y máximo de la cuenca
        self.area_min = cfg_camels['area'].get('min', 100)  # km²
        self.area_max = cfg_camels['area'].get('max', None)  # km²

        # Disponibilidad mínima de datos durante el periodo de estudio
        self.disponibilidad = cfg_camels['caudal'].get('disponibilidad', .8)
        self.min_años = cfg_camels['caudal'].get('n_años', 10)

        # Sistema de coordenadas
        self.crs = self.cfg.get('crs', 4326)
