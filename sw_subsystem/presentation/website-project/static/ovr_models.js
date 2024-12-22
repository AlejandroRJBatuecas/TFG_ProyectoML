// Creación de tablas de OvR models
document.addEventListener('DOMContentLoaded', () => {
    const colorMap = {}; // Mapa para almacenar colores asignados a los modelos
    const predefinedColors = [
        '#FF2D00', '#00FF2D', '#002DFF', '#F3FF00', '#33FFF5', '#F5FF33', '#FF8C33', '#33FF8C', '#8C33FF'
    ]; // Lista fija de colores reutilizables
    let colorIndex = 0;
  
    // Función para obtener o asignar un color a un modelo
    function getModelColor(modelName) {
        if (!colorMap[modelName]) {
            colorMap[modelName] = predefinedColors[colorIndex % predefinedColors.length];
            colorIndex++;
        }
        return colorMap[modelName];
    }
  
    // Iterar sobre cada canvas para los patrones
    document.querySelectorAll('canvas[id^="chart-pattern-"]').forEach(canvas => {
        const patternId = canvas.id.replace('chart-pattern-', ''); // Extraer el patrón del ID del canvas
        const rows = document.querySelectorAll(`div[id^="ovr_models_performance_comparison-pattern-${patternId}-model-"]`); // Filtrar filas de datos por patrón
  
        const chartData = {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1'], // Métricas
            datasets: []
        };
  
        rows.forEach(row => {
            const modelName = row.id.replace(`ovr_models_performance_comparison-pattern-${patternId}-model-`, '').replace('-with-best-features', ' (Best Features)').replace('-row', ''); // Extraer el nombre del modelo
            const cells = row.querySelectorAll('.col-2 span'); // Seleccionar celdas de métricas
  
            const metrics = Array.from(cells).map(cell =>
                parseFloat(cell.textContent.replace('%', '').trim()) // Extraer y limpiar el valor de la celda
            );
  
            // Añadir los datos al dataset
            chartData.datasets.push({
                label: modelName,
                data: metrics,
                backgroundColor: getModelColor(modelName),
                borderColor: getModelColor(modelName),
                borderWidth: 1
            });
        });
  
        // Crear la gráfica con Chart.js
        const pattern_chart = new Chart(canvas, {
            type: 'bar',
            data: chartData,
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top'
                    },
                    tooltip: {
                        callbacks: {
                            label: function (tooltipItem) {
                                return `${tooltipItem.dataset.label}: ${tooltipItem.raw.toFixed(2)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Model performance metrics'
                        }
                    },
                    y: {
                        beginAtZero: false, // No comienza en 0
                        min: 97, // Valor mínimo del eje Y
                        title: {
                            display: true,
                            text: 'Value (%)'
                        }
                    }
                }
            }
        });
    });
});