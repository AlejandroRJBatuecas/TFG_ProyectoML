const bootstrap_columns_num = 12; // Número máximo de columnas para una fila

const metrics_body = document.getElementById('metrics-body');

let metrics_json = null;
let circuit_count = 0;
let circuits = [];
let metrics_column_count = 0;

// Encabezado de la tabla de métricas
const header_row = document.getElementById('metrics-header-row');

function resizeColumns() {
    const metrics_columns_total = circuit_count + 1; // Número de circuitos 
    const metrics_columns_width = Math.floor(bootstrap_columns_num/metrics_columns_total);

    console.log(metrics_columns_width);

    const header_row_children = header_row.children;
    const colClassPattern = /^col-\d+$/;

    for (children of header_row_children) {
        const classList = children.classList;

        // Buscar la clase que coincide con el patrón `col-x`
        for (let i = 0; i < classList.length; i++) {
            if (colClassPattern.test(classList[i])) {
                const currentClass = classList[i];
                console.log(currentClass);
                // Asegurarse de que el nuevo número de columna esté dentro de un rango válido (por ejemplo, de 1 a 12)
                if (metrics_columns_width >= 1 && metrics_columns_width <= 12) {
                    const newClass = `col-${metrics_columns_width}`;
                    classList.replace(currentClass, newClass);
                }
                break;
            }
        }
    }
}

function resizeTable() {
    // Redimensionar columnas
    resizeColumns();
}

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
            console.log('El elemento no existe.');
        }
    } else {
        console.log('No se encontró ningún número en la cadena.');
    }

}

function findInsertionPosition() {
    let extended_circuits = [true, ...circuits, false]; // Array auxiliar para buscar la posición donde insertar el nuevo circuito
    
    // Buscar el índice de la primera posición libre cuya posición anterior esté ocupada, para añadir el circuito en esa posición
    // Restar 1 al final, ya que se ha insertado un true al principio del array auxiliar
    return extended_circuits.findIndex((_, index) => extended_circuits[index] == false && extended_circuits[index-1] == true) - 1
}

function generateCircuitColumn() {
    // Incrementar el contador de circuitos
    circuit_count += 1;

    // Buscar el número de circuito a añadir
    let insertion_position = findInsertionPosition();
    let circuit_number = insertion_position+1;

    // Crear la columna del circuito
    const circuit_column = document.createElement('div');
    circuit_column.classList.add('col-2', 'text-white', 'text-center', 'bg-info');
    circuit_column.id = "metrics-circuit-"+circuit_number+"-column";

    // Crear y añadir el encabezado del circuito
    const circuit_header = `
        <div class="row justify-content-around text-white bg-secondary border border-dark" id="metrics-circuit-${circuit_number}-header">
            <div class="col-12 p-2 metric-col-height d-flex justify-content-center align-items-center">
                <button type="button" class="btn btn-danger w-100 full-height" id="metrics-remove-circuit-${circuit_number}-button">
                    <!-- Nombre del botón de eliminar circuito abreviado para pantallas pequeñas -->
                    <span class="d-md-none pe-none">X</span>
                    <!-- Nombre del botón de eliminar circuito completo para pantallas grandes -->
                    <span class="d-none d-md-block pe-none">Eliminar circuito</span>
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
                <div class="col-12 p-2 border border-dark bg-primary">
                    <!-- Nombre de circuito abreviado para pantallas pequeñas -->
                    <h4 class="mb-0 text-truncate d-md-none">C${circuit_number}</h4>
                    <!-- Nombre de circuito completo para pantallas grandes -->
                    <h4 class="mb-0 text-truncate d-none d-md-block">Circuito ${circuit_number}</h4>
                </div>
            `;
            metric_category_header.insertAdjacentHTML("afterbegin", metric_category_header_content)

            circuit_column.appendChild(metric_category_header);

            const metrics = metrics_json[category];

            for (const metric in metrics) {
                if (metrics.hasOwnProperty(metric)) {
                    const metric_row = document.createElement('div');
                    metric_row.classList.add('row');
                    metric_row.id = "metrics-metric-"+metric+"-circuit-"+circuit_number+"-row";

                    const metric_row_content = `
                        <div class="col-12 p-2 metric-col-height text-truncate border border-dark d-flex justify-content-center align-items-center">
                            <input type="text" class="form-control" id="${metric}-circuit-${circuit_number}-value" name="${metric}-circuit-${circuit_number}-value" value="${metrics[metric]}">
                        </div>
                    `;
                    metric_row.insertAdjacentHTML("afterbegin", metric_row_content)

                    circuit_column.appendChild(metric_row);
                }
            }
        }
    }

    // Habilitar la posición del circuito
    circuits[insertion_position] = true;

    // Obtener la columna anterior donde se va a insertar
    let previous_column = document.getElementById('metrics-circuit-'+(circuit_number-1)+'-column');

    if (insertion_position == 0) { // Si es la primera posición, insertar al principio (tras la columna de métricas)
        // Obtener la columna de métricas
        previous_column = document.getElementById('metrics-category-column-'+metrics_column_count);
    }

    // Insertar el nuevo circuito después de la columna anterior
    previous_column.insertAdjacentElement("afterend", circuit_column);

    // Añadir el evento del botón Eliminar Circuito
    document.getElementById('metrics-remove-circuit-'+circuit_number+'-button').addEventListener('click', eliminateCircuitColumn);
}

function generateMetricsColumn() {
    // Incrementar el contador de columnas de métricas
    metrics_column_count += 1;

    // Crear la columna de las métricas
    const metrics_column = document.createElement('div');
    metrics_column.classList.add('col-4', 'text-white', 'text-center', 'bg-info');
    metrics_column.id = "metrics-category-column-"+metrics_column_count;

    // Crear y añadir el encabezado del circuito
    const metrics_header = `
        <div class="row justify-content-around text-white bg-secondary border border-dark" id="metrics-header-row">
            <div class="col-12 p-2 metric-col-height d-flex justify-content-center align-items-center">
                <h3 class="mb-0">Métricas</h3>
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
                <div class="col-12 p-2 border border-dark text-truncate bg-primary">
                    <h4 class="mb-0 text-truncate" title="${category}">${category}</h4>
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
                        <div class="col-12 metric-col-height border border-dark d-flex justify-content-center align-items-center">
                            <h5 class="mb-0">${metric.replace(/m\./g, "")}</h5>
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

// Obtener los datos de las métricas del JSON desde el servidor
async function getMetrics() {
    const response = await fetch('/obtener_metricas');
    metrics_json = await response.json();
    generateMetricsColumn()
    generateCircuitColumn();
}

// Llamar a la función para obtener los datos cuando la página cargue
window.onload = getMetrics;

// Evento para añadir un nuevo circuito
document.getElementById('metrics-add-circuit-button').addEventListener('click', generateCircuitColumn);