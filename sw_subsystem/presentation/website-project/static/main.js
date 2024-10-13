function adjustMarginTop() {
    const navbarHeight = document.getElementById('main-navbar').offsetHeight;
    const fixedDivHeight = document.getElementById('fixed-content').offsetHeight;
    const totalHeight = navbarHeight + fixedDivHeight;

    // Ajustar el margin-top del contenido principal
    document.getElementById('metrics-container').style.marginTop = totalHeight + 'px';
}

// Llamar a la función cuando cargue la página con el contenido fijo
window.addEventListener('load', function () {
    if (document.getElementById('fixed-content')) {
        adjustMarginTop();
    }
})

// Habilitación de los tooltips
var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
  return new bootstrap.Tooltip(tooltipTriggerEl)
})

// Habilitación de los popovers
var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'))
var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
  return new bootstrap.Popover(popoverTriggerEl)
})