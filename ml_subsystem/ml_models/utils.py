import matplotlib.pyplot as plt
import tkinter as tk

from pathlib import Path

# Crear carpeta de imágenes
IMAGES_PATH = Path(__file__).resolve().parent.parent / "images"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

# Obtener la resolución de la pantalla
def get_screen_resolution():
    # Obtener dimensiones de la pantalla en píxeles
    root = tk.Tk()
    screen_width_pixels = root.winfo_screenwidth()
    screen_height_pixels = root.winfo_screenheight()
    root.withdraw()

    # Convertir de píxeles a pulgadas (asumiendo 96 PPI)
    ppi = 96
    screen_width_inches = screen_width_pixels / ppi
    screen_height_inches = screen_height_pixels / ppi

    return screen_width_inches, screen_height_inches, ppi

# Crear una figura para mostrarla en pantalla completa
def create_full_screen_figure():
    screen_width_inches, screen_height_inches, ppi = get_screen_resolution()

    # Modificar las configuraciones preterminadas de Matplotlib
    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('figure', figsize=(screen_width_inches, screen_height_inches))
    plt.rc('figure', dpi=ppi)

    # Mostrar la figura en pantalla completa con plt.show()
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')  # Maximizar la ventana

# Guardar una figura en la carpeta images
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    print(f"\nGuardando figura en {path}")  # <-- Depuración
    plt.savefig(path, format=fig_extension, dpi=resolution)
    plt.close()
    print(f"Imagen guardada en: {IMAGES_PATH / f'{fig_id}.{fig_extension}'}")

# Crear el conjunto de prueba de manera manual
def create_test_set(data, train_ratio):
    # Obtener el número de proyectos correspondiente al 70%
    train_set_index = int(len(data.groupby('project'))*train_ratio)
    print("Índice del proyecto 70%: ", train_set_index)

    # Obtener el número de fila correspondiente a los proyectos
    train_set_row = data.groupby('project').head(1).index[train_set_index]
    print("Fila del proyecto 70%: ", train_set_row)

    return data.iloc[:train_set_row], data.iloc[train_set_row:]
