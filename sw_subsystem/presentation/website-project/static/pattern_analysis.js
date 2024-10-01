const bootstrap_columns_num = 12; // Número máximo de columnas para una fila

// Formulario de la tabla de métricas
const formulario = document.getElementById('metrics-form');
// Cuerpo de la tabla de métricas
const metrics_body = document.getElementById('metrics-body');
// Encabezado de la tabla de métricas
const header_row = document.getElementById('metrics-header-row');

let metrics_json = null;
let circuit_count = 0;
let circuits = [];

function eliminateCircuitColumn(event) {
    // Obtener el id del elemento clicado
    const clicked_element_id = event.target.id;

    // Usar una expresión regular para encontrar el número en la cadena
    const match = clicked_element_id.match(/\d+/);

    // Si se encuentra un número, se extrae. Si no, se maneja el caso en que no se encuentre
    if (match) {
        const circuit_number = match[0];

        // Eliminar la columna del circuito
        const circuit_column = document.getElementById('metrics-circuit-'+circuit_number+'-column');
            
        // Verificar si la columna existe antes de eliminarla
        if (circuit_column) {
            // Eliminar la columna del DOM
            circuit_column.remove();
            if (circuit_number == circuits.length) {
                circuits.pop()
            } else {
                // Desactivar la posición del circuito del array
                circuits[circuit_number-1] = false;
            }
            // Decrementar el contador de circuitos
            circuit_count -= 1;
        } else {
            console.log('The element does not exist');
        }
    } else {
        console.log('No number was found in the string');
    }

}

function findInsertionPosition() {
    let extended_circuits = [true, ...circuits, false]; // Array auxiliar para buscar la posición donde insertar el nuevo circuito
    
    // Buscar el índice de la primera posición libre cuya posición anterior esté ocupada, para añadir el circuito en esa posición
    // Restar 1 al final, ya que se ha insertado un true al principio del array auxiliar
    return extended_circuits.findIndex((_, index) => !extended_circuits[index] && extended_circuits[index-1]) - 1
}

function addFormInputs(metrics, circuit_number, circuit_metrics, circuit_column) {
    for (const metric in metrics) {
        if (metrics.hasOwnProperty(metric)) {
            const metric_row = document.createElement('div');
            metric_row.classList.add('row');
            metric_row.id = "metrics-metric-"+metric+"-circuit-"+circuit_number+"-row";

            // Generar los inputs por defecto
            let metric_row_content = `
                <div class="col p-1 input-col-height text-truncate border border-dark d-flex justify-content-center align-items-center">
                    <input type="text" class="form-control-sm w-75" id="${metric}_circuit_${circuit_number}_value" name="${metric}_circuit_${circuit_number}_value" value="${metrics[metric]}" required="required">
                </div>
            `;

            // Si se han obtenido datos del json, modificar el value de los inputs
            if (circuit_metrics?.circuit_metrics[metric]) { // circuit_metrics?.circuit_metrics[metric] == circuit_metrics && circuit_metrics[metric]
                
                metric_row_content = `
                    <div class="col p-1 input-col-height text-truncate border border-dark d-flex justify-content-center align-items-center">
                        <input type="text" class="form-control-sm w-75" id="${metric}_circuit_${circuit_number}_value" name="${metric}_circuit_${circuit_number}_value" value="${circuit_metrics[metric]}" required="required">
                    </div>
                `;
            }
            
            metric_row.insertAdjacentHTML("afterbegin", metric_row_content)

            circuit_column.appendChild(metric_row);
        }
    }
}

function generateCircuitColumn(circuit_metrics = null) {
    // Verificamos si circuit_metrics es realmente un evento al ponerlo como primer parámetro
    if (circuit_metrics instanceof PointerEvent) {
        // Si es un evento, lo tratamos como que no hay circuit_metrics
        circuit_metrics = null;
    }

    // Incrementar el contador de circuitos
    circuit_count += 1;

    // Buscar el número de circuito a añadir
    let insertion_position = findInsertionPosition();
    let circuit_number = insertion_position+1;

    // Crear la columna del circuito
    const circuit_column = document.createElement('div');
    circuit_column.classList.add('col', 'text-center');
    circuit_column.id = "metrics-circuit-"+circuit_number+"-column";

    // Crear y añadir el encabezado del circuito
    const circuit_header = `
        <div class="row justify-content-around text-white bg-secondary border border-dark" id="metrics-circuit-${circuit_number}-header">
            <div class="col p-2 metric-col-height d-flex justify-content-center align-items-center">
                <button type="button" class="btn btn-danger btn-sm w-100 full-height" id="metrics-remove-circuit-${circuit_number}-button">
                    <!-- Nombre del botón de eliminar circuito abreviado para pantallas pequeñas -->
                    <span class="d-md-none pe-none">X</span>
                    <!-- Nombre del botón de eliminar circuito completo para pantallas grandes -->
                    <span class="d-none d-md-block pe-none">Delete circuit</span>
                </button>
                
            </div>
        </div>
    `;
    circuit_column.insertAdjacentHTML("afterbegin", circuit_header);

    // Añadir cada una de las métricas y sus categorías
    for (const category in metrics_json) {
        if (metrics_json.hasOwnProperty(category)) {
            const metric_category_header = document.createElement('div');
            metric_category_header.classList.add('row');
            metric_category_header.id = "metrics-category-"+category+"-circuit-"+circuit_number+"-header";

            const metric_category_header_content = `
                <div class="col p-2 border border-dark bg-primary text-white text-truncate">
                    <!-- Nombre de circuito abreviado para pantallas pequeñas -->
                    <span class="mb-0 fw-bold d-md-none" title="C${circuit_number}">C${circuit_number}</span>
                    <!-- Nombre de circuito completo para pantallas grandes -->
                    <span class="mb-0 fw-bold d-none d-md-block" title="Circuit ${circuit_number}">Circuit ${circuit_number}</span>
                </div>
            `;
            metric_category_header.insertAdjacentHTML("afterbegin", metric_category_header_content)

            circuit_column.appendChild(metric_category_header);

            const metrics = metrics_json[category];

            addFormInputs(metrics, circuit_number, circuit_metrics, circuit_column);
        }
    }

    // Habilitar la posición del circuito
    circuits[insertion_position] = true;

    // Obtener la columna anterior donde se va a insertar
    let previous_column = document.getElementById('metrics-circuit-'+(circuit_number-1)+'-column');

    if (insertion_position == 0) { // Si es la primera posición, insertar al principio (tras la columna de métricas)
        // Obtener la columna de métricas
        previous_column = document.getElementById('metrics-category-column');
    }

    // Insertar el nuevo circuito después de la columna anterior
    previous_column.insertAdjacentElement("afterend", circuit_column);

    // Añadir el evento del botón Eliminar Circuito
    document.getElementById('metrics-remove-circuit-'+circuit_number+'-button').addEventListener('click', eliminateCircuitColumn);
}

function generateMetricsColumn() {
    // Crear la columna de las métricas
    const metrics_column = document.createElement('div');
    metrics_column.classList.add('col-4', 'sticky', 'text-center');
    metrics_column.id = "metrics-category-column";

    // Crear y añadir el encabezado del circuito
    const metrics_header = `
        <div class="row justify-content-around text-white bg-secondary border border-dark" id="metrics-header-row">
            <div class="col p-2 metric-col-height d-flex justify-content-center align-items-center">
                <h5 class="mb-0 fw-bold text-truncate">Metrics</h5>
            </div>
        </div>
    `;
    metrics_column.insertAdjacentHTML("afterbegin", metrics_header);

    // Añadir cada una de las métricas y sus categorías
    for (const category in metrics_json) {
        if (metrics_json.hasOwnProperty(category)) {
            const metric_category_header = document.createElement('div');
            metric_category_header.classList.add('row');
            metric_category_header.id = "metrics-category-"+category+"-header";

            const metric_category_header_content = `
                <div class="col p-2 border border-dark bg-primary text-white text-truncate">
                    <span class="mb-0 fw-bold" title="${category}">${category}</span>
                </div>
            `;
            metric_category_header.insertAdjacentHTML("afterbegin", metric_category_header_content)

            metrics_column.appendChild(metric_category_header);

            const metrics = metrics_json[category];

            for (const metric in metrics) {
                if (metrics.hasOwnProperty(metric)) {
                    const metric_row = document.createElement('div');
                    metric_row.classList.add('row');
                    metric_row.id = "metrics-metric-"+metric+"-row";

                    const metric_row_content = `
                        <div class="col p-1 input-col-height border border-dark d-flex justify-content-center align-items-center text-truncate bg-info">
                            <span class="mb-0 fw-bold" title="${metric.replace(/m\./g, "")}">${metric.replace(/m\./g, "")}</span>
                        </div>
                    `;
                    metric_row.insertAdjacentHTML("afterbegin", metric_row_content)

                    metrics_column.appendChild(metric_row);
                }
            }
        }
    }

    // Añadir la columna de las métricas al cuerpo de la tabla
    metrics_body.appendChild(metrics_column);
}

function getFileExtension(file_type) {
    if (file_type == "json") {
        return ".json"
    }
    return file_type == "python" ? ".py" : file_type
}

function openFileImportModal(file_type) {
    // Modal para la importación de fichero
    const file_import_modal = new bootstrap.Modal(document.getElementById('file-import-modal'));

    // Span para modificar la extensión del fichero requerida
    const file_extension_type_span = document.getElementById('file-extension-type');
    file_extension_type_span.innerText = file_type

    // Obtener el input del fichero
    const file_input = document.getElementById('file-input');
    // Obtener la extensión del fichero
    const file_extension = getFileExtension(file_type)
    // Modificar el accept del input del fichero
    file_input.accept = file_extension

    // Modificar el accept del input del fichero
    const file_extension_allowed = document.getElementById('file-extension-allowed')
    file_extension_allowed.value = file_extension

    // Mostrar el modal
    file_import_modal.show();
}

// Obtener los datos de las métricas del JSON desde el servidor
async function getMetrics() {
    const response = await fetch('/get_metrics');
    metrics_json = await response.json();

    // Generar la columna de las métricas
    generateMetricsColumn();

    const circuit_metrics_div = document.getElementById('circuit-metrics');
    const circuits_list_div = document.getElementById('circuits-list');

    if (circuit_metrics_div) { // Si se están obteniendo datos de un archivo JSON, modificar los circuitos y valores
        // Obtener los datos del atributo "circuit-metrics-data" y los parseamos como JSON
        const circuit_metrics = JSON.parse(circuit_metrics_div.getAttribute('circuit-metrics-data'));
        // Generar una columna por circuito
        for (const circuit in circuit_metrics) {
            generateCircuitColumn(circuit_metrics[circuit])
        } 
    } else if (circuits_list_div) { // Si se están obteniendo datos de un archivo python, modificar los circuitos y valores
        // Obtener los datos del atributo "circuits-list-data" y los parseamos como JSON
        const circuits_list = JSON.parse(circuits_list_div.getAttribute('circuits-list-data'));
        // Generar una columna por circuito
        for (const circuit in circuits_list) {
            generateCircuitColumn(circuits_list[circuit])
        } 
    } else { // Si no, generar la columna de circuito por defecto
        generateCircuitColumn();
    }
}

// Llamar a la función para obtener los datos cuando la página cargue
window.onload = getMetrics;

// Evento para añadir un nuevo circuito
document.getElementById('metrics-add-circuit-button').addEventListener('click', generateCircuitColumn);

// Evento para abrir el modal con el botón Importar fichero JSON
document.getElementById('json-file-import-button').addEventListener('click', function() {
    openFileImportModal("json");
});

// Evento para abrir el modal con el botón Importar fichero de código
document.getElementById('code-file-import-button').addEventListener('click', function() {
    openFileImportModal("python");
});

// Evento submit del formulario, para la validación de los campos numéricos
formulario.addEventListener('submit', function(event) {
    // Prevenir el envío por defecto
    event.preventDefault();

    // Obtener todos los inputs del formulario
    const inputs = formulario.querySelectorAll('input[type="text"]');
    const checkboxes = document.querySelectorAll('input[type="checkbox"]');

    // Variable para verificar si todos los campos son válidos
    let is_valid = true;
    let not_valid_inputs = [];

    // Variable para verificar si existe algún checkbox marcado
    let checkbox_checked = false;

    // Recorrer cada input y validar si es un número
    inputs.forEach(function(input) {
        const valor = input.value.trim(); // Obtener el valor del input y quitar espacios en blanco

        // Validar si el valor es un número
        if (valor === '' || isNaN(valor)) {
            is_valid = false;
            input.style.borderColor = 'red'; // Marcar el input en rojo si no es válido
            input.classList.add('border', 'border-danger', 'border-3');
            not_valid_inputs.push(input.name);
        } else {
            input.style.borderColor = ''; // Restablecer el borde si es válido
            input.classList.remove('border', 'border-danger', 'border-3');
        }
    });

    // Comprobar si al menos un checkbox está marcado
    checkboxes.forEach(function(checkbox) {
        if (checkbox.checked) {
            checkbox_checked = true;
        }
    });

    // Si todos los inputs son válidos, enviar el formulario
    if (is_valid && checkbox_checked) {
        formulario.submit();
    } else {
        let title_string = "";
        let error_string = "";

        // Obtener los elementos de texto para mostrar el error
        const error_info_title = document.getElementById('error-info-title');
        const error_info_text = document.getElementById('error-info-text');

        if (!checkbox_checked) {
            // Establecer el título y el texto del error
            title_string = "No model selected";
            error_string = "You must select at least one of the proposed ML models";
        } else {
            // Establecer el título del error
            title_string = "The values ​​for the following metrics must be numeric:";

            // Obtener todos los campos no válidos y añadirlos al mensaje de error
            not_valid_inputs.forEach(function(not_valid_input) {
                const splitted_name = not_valid_input.split("_");
                // splitted_name[0] contiene el nombre de la métrica y splitted_name[2] contiene el número del circuito
                error_string += 'Circuit '+splitted_name[2]+' - '+splitted_name[0].replace("m.", "")+'\n'
            });
        }

        // Establecer los mensajes de error
        error_info_title.innerText = title_string;
        error_info_text.innerText = error_string;

        // Modal para la información de errores en el formulario
        const error_info_modal = new bootstrap.Modal(document.getElementById('error-info-modal'));

        // Mostrar el modal
        error_info_modal.show();
    }
});