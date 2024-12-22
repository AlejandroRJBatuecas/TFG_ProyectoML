// Plugin para mostrar texto en el centro
const centerTextPlugin = {
    id: 'centerText',
    beforeDraw: function (chart) {
        const { width } = chart;
        const { height } = chart;
        const ctx = chart.ctx;
        ctx.save();

        let percentage = chart.data.datasets[0].data[0];
        percentage = typeof percentage === 'number' ? percentage : parseFloat(percentage);
        const formattedPercentage = percentage.toFixed(2);

        const fontSize = Math.min(width, height) / 10;
        ctx.font = `${fontSize}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#000';

        const textX = width / 2;
        const textY = height / 2;

        ctx.fillText(`${formattedPercentage}%`, textX, textY);
        ctx.restore();
    },
};

// Configuración común para las gráficas
const commonConfig = {
    type: 'doughnut',
    options: {
        responsive: false,
        maintainAspectRatio: false, // Ignora la relación de aspecto si es necesario
        plugins: {
        tooltip: { enabled: false }, // Desactiva tooltips
        },
    },
    plugins: [centerTextPlugin],
};

// Función para crear gráficas dinámicamente
function createCharts() {
    // Seleccionar todos los canvas con la clase .chart-dimensions
    const canvases = document.querySelectorAll('.chart-dimensions');

    canvases.forEach((canvas) => {
        // Leer el valor decimal desde el atributo data-value
        const rawValue = canvas.getAttribute('data-value');
        const value = parseFloat(rawValue); // Convertir a número decimal

        const ctx = canvas.getContext('2d');

        // Crear la gráfica
        const model_performance_metrics_chart = new Chart(ctx, {
        ...commonConfig,
        data: {
            datasets: [
            {
                data: [value, (100 - value).toFixed(2)], // Asegura que los valores sean numéricos
                backgroundColor: ['#4CAF50', '#e0e0e0'], // Colores
                borderWidth: 0,
            },
            ],
        },
        });
    });
}

// Llama a la función para pintar las gráficas
createCharts();