let metrics_json = null;
let circuit_count = 1;

// Función para generar el formulario dinámicamente
function generateForm(metrics_json) {
    const formContainer = document.getElementById('metrics-form');

    // Crear un div con las clases especificadas
    const formWrapper = document.createElement('div');
    formWrapper.classList.add('col-12', 'col-md-6', 'p-1');

    const circuit_name = document.createElement('h2');
    circuit_name.innerText = "Circuito "+circuit_count
    formWrapper.appendChild(circuit_name)

    const form = document.createElement('form');
    const metrics_row = document.createElement('div');
    metrics_row.classList.add('row', 'justify-content-center', 'mt-3', 'div-bg-color', 'p-3')
    form.appendChild(metrics_row)

    for (const key in metrics_json) {
        if (metrics_json.hasOwnProperty(key)) {
            const metric_col = document.createElement('div');
            metric_col.classList.add('col-12')

            const label = document.createElement('label');
            label.htmlFor = key;
            label.innerText = key;

            const input = document.createElement('input');
            input.type = 'text';
            input.id = key;
            input.name = key;
            input.value = metrics_json[key];
            input.classList.add('form-control');

            metric_col.appendChild(label);
            metric_col.appendChild(input);
            metrics_row.appendChild(metric_col)
        }
    }

    formWrapper.appendChild(form);
    formContainer.appendChild(formWrapper);
}

// Función para obtener los datos del JSON desde el backend
async function fetchFormData() {
    const response = await fetch('/obtener_metricas');
    metrics_json = await response.json();
    generateForm(metrics_json);
}

// Llamada a la función para obtener los datos y generar el formulario cuando la página cargue
window.onload = fetchFormData;