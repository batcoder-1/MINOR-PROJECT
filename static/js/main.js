$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        if (!$('#imageUpload').val()) {
            $('#result').fadeIn(600);
            $('#result').text('Please choose an image before prediction.');
            return;
        }

        var form_data = new FormData($('#upload-file')[0]);
        var predictButton = $(this);

        // Show loading animation
        predictButton.prop('disabled', true).text('Predicting...');
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('#result').fadeIn(600);
                $('#result').text(data);
                console.log('Success!');
            },
            error: function (xhr) {
                var errorMessage = 'Prediction failed. Please try another image.';
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    errorMessage = xhr.responseJSON.error;
                }
                $('#result').fadeIn(600);
                $('#result').text(errorMessage);
            },
            complete: function () {
                $('.loader').hide();
                predictButton.prop('disabled', false).text('Predict');
            }
        });
    });

});