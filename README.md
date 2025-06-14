## QPP-ML: Herramienta de machine learning para la detección y predicción de patrones de diseño en software cuántico

### Requisitos
- Python 3.8 o superior
- `virtualenv` (puedes instalarlo con `pip install virtualenv`)

### Instalación
Sigue estos pasos para clonar el repositorio y configurar el entorno virtual:

```bash
# Clona el repositorio
git clone https://github.com/AlejandroRJBatuecas/TFG_ProyectoML.git

# Accede a la raíz del proyecto
cd TFG_ProyectoML

# Crea un entorno virtual de la web
python -m venv ./sw_subsystem/presentation/website-env

# Activa el entorno virtual
# En Windows:
.\sw_subsystem\presentation\website-env\Scripts\activate

# En macOS/Linux:
source ./sw_subsystem/presentation/bin/activate

# Instala las dependencias
pip install -r requirements.txt
```

### Uso
1º. Activa el entorno virtual (si no lo está)
```bash
# En Windows:
.\sw_subsystem\presentation\website-env\Scripts\activate

# En macOS/Linux:
source ./sw_subsystem/presentation/bin/activate
```

2º. Ejecuta el script de arranque
```bash
# En Windows:
.\run_script.bat
```

**Al ejecutar el arranque, si algunos de los modelos de ML no está entrenado o no encuentra, se entrenará automáticamente antes de iniciarse la aplicación.**
- El tiempo requerido dependerá del modelo y su tipo de entrenamiento
- El proceso y sus estadísticas de rendimiento se mostrarán a través de la consola durante el entrenamiento