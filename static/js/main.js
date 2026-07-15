// main.js - Credit Shield AI Client Validation & Spinner Loaders

(function () {
    'use strict';

    // Fetch the form we want to apply custom Bootstrap validation styles to
    const forms = document.querySelectorAll('.needs-validation');

    // Loop over them and prevent submission if invalid
    Array.prototype.slice.call(forms)
        .forEach(function (form) {
            form.addEventListener('submit', function (event) {
                if (!form.checkValidity()) {
                    event.preventDefault();
                    event.stopPropagation();
                    
                    // Reset spinner if validation fails
                    const spinner = form.querySelector('.btn-spinner');
                    if (spinner) {
                        spinner.classList.add('d-none');
                    }
                } else {
                    // If form is valid, trigger spinner state
                    const spinner = form.querySelector('.btn-spinner');
                    const submitBtn = form.querySelector('button[type="submit"]');
                    if (spinner && submitBtn) {
                        spinner.classList.remove('d-none');
                        submitBtn.setAttribute('disabled', 'true');
                    }
                }

                form.classList.add('was-validated');
            }, false);
        });
})();
