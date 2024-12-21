function set_nav_bar_active_element() {
    // Obtener la URL actual de la página
    const path = window.location.pathname;

    // Extraer el nombre del archivo actual
    const page = path.split("/").pop();

    // Seleccionar todos los elementos de la lista de navegación
    const navLinks = document.querySelectorAll('.nav-link');

    // Recorrer cada enlace y verificar si el href coincide con la página actual
    navLinks.forEach(function(link) {
        if (link.getAttribute('href')) {
            const linkHref = link.getAttribute('href').replace("/", "");

            // Si el href coincide con la página actual, añadir la clase 'active-custom'
            if (linkHref === page) {
                link.classList.add('active-custom');
            }
        }
    });
}

function adjustMarginTop() {
    const navbarHeight = document.getElementById('main-navbar').offsetHeight;
    const fixedDivHeight = document.getElementById('fixed-content').offsetHeight;
    const totalHeight = navbarHeight + fixedDivHeight;

    // Ajustar el margin-top del contenido principal
    document.getElementById('metrics-container').style.marginTop = totalHeight + 'px';
}

// Llamar a la función cuando cargue la página con el contenido fijo
window.addEventListener('load', function () {
    set_nav_bar_active_element()

    if (document.getElementById('fixed-content')) {
        adjustMarginTop();
    }
})

// Habilitación de los tooltips
const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
  return new bootstrap.Tooltip(tooltipTriggerEl)
})

// Habilitación de los popovers
const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'))
const popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
  return new bootstrap.Popover(popoverTriggerEl)
})