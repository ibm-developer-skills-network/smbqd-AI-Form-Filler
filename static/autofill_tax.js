function typeEffect(element, text, delay) {
    let index = 0;

    function typeCharacter() {
        if (index < text.length) {
            element.value += text.charAt(index);
            index++;
            setTimeout(typeCharacter, delay);
        }
    }

    typeCharacter();
}

function autoFillTaxForm() {
    // Show the loading spinner
    document.getElementById('loadingSpinner').style.display = 'flex';

    fetch(window.location.origin+'/api/get_tax_form_data')
        .then(response => response.json())
        .then(data => {
            // Clear existing values
            document.getElementById('fullName').value = '';
            document.getElementById('ssn').value = '';
            document.getElementById('income').value = '';
            document.getElementById('deductions').value = '';
            document.getElementById('taxPaid').value = '';
            document.getElementById('descriptions_request_for_extra_tax_return').value = '';

            // Apply typing effect to each field with a delay
            typeEffect(document.getElementById('fullName'), data.fullName, 100);
            setTimeout(() => typeEffect(document.getElementById('ssn'), data.ssn, 100), 1000);
            setTimeout(() => typeEffect(document.getElementById('income'), data.income, 100), 2000);
            setTimeout(() => typeEffect(document.getElementById('deductions'), data.deductions, 100), 3000);
            setTimeout(() => typeEffect(document.getElementById('taxPaid'), data.taxPaid, 100), 4000);
            setTimeout(() => typeEffect(document.getElementById('descriptions_request_for_extra_tax_return'), data.descriptions_request_for_extra_tax_return, 100), 5000);

            // Hide the loading spinner after all fields are filled
            setTimeout(() => {
                document.getElementById('loadingSpinner').style.display = 'none';
            }, 5000);

            console.log('Tax form has been auto-filled with:', data);
        })
        .catch(error => {
            console.error('Error fetching tax form data:', error);
            document.getElementById('loadingSpinner').style.display = 'none';
        });
}

// You may include additional JavaScript functions or code here if needed
