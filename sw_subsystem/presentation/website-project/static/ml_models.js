document.addEventListener('DOMContentLoaded', () => {
    // Seleccionar las filas que contienen los datos de rendimiento de los modelos
    const performanceRows = document.querySelectorAll('[id^="ml_models_performance_comparison-model-"][id$="-row"]');
  
    // Inicializar las métricas y datos por modelo
    let models = [];
    const metrics = ['Accuracy', 'Precision', 'Recall', 'F1'];
    const dataByModel = {};
  
    // Extraer los datos de la tabla y organizar por modelo y métrica
    performanceRows.forEach(row => {
      const modelName = row.querySelector('.col-4 span').textContent.trim();
      const metricCells = row.querySelectorAll('.col-2 span');
  
      models.push(modelName);
      dataByModel[modelName] = metrics.map((_, index) =>
        parseFloat(metricCells[index].textContent.replace('%', ''))
      );
    });
  
    // Crear datasets para Chart.js
    const datasets = Object.keys(dataByModel).map((model, index) => ({
      label: model, // Nombre del modelo
      data: dataByModel[model], // Valores de las métricas
      backgroundColor: `rgba(${50 + index * 50}, ${100 + index * 30}, ${200 - index * 40}, 0.6)`,
      borderColor: `rgba(${50 + index * 50}, ${100 + index * 30}, ${200 - index * 40}, 1)`,
      borderWidth: 1
    }));
  
    // Generar la gráfica de barras verticales
    const ctx = document.getElementById('verticalBarChart').getContext('2d');
    const verticalBarChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: metrics, // Agrupaciones principales (Métricas)
        datasets: datasets // Barras por modelo dentro de cada métrica
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            position: 'top', // Posición de la leyenda
          },
          tooltip: {
            callbacks: {
              label: function (tooltipItem) {
                return `${tooltipItem.dataset.label}: ${tooltipItem.raw}%`; // Mostrar modelo y valor en el tooltip
              }
            }
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Model performance metrics' // Título del eje X
            }
          },
          y: {
            beginAtZero: false, // No comienza en 0
            min: 85, // Valor mínimo del eje Y
            title: {
              display: true,
              text: 'Value (%)' // Título del eje Y
            }
          }
        }
      }
    });
});