$(document).ready(function () {
    var imageSection = $('.image-section');
    var loader = $('.loader');
    var result = $('#result');
    var imagePreview = $('#imagePreview');
    var predictButton = $('#btn-predict');
    var fileName = $('#fileName');
    var dropzone = $('#dropzone');

    imageSection.hide();
    loader.hide();
    result.hide();

    function showResult(message, isError) {
        result.removeClass('is-error');
        if (isError) {
            result.addClass('is-error');
        }

        result.stop(true, true).hide().text(message).fadeIn(350);
    }

    function readURL(input) {
        if (input.files && input.files[0]) {
            var selectedFile = input.files[0];
            var reader = new FileReader();

            fileName.text(selectedFile.name);

            reader.onload = function (e) {
                imagePreview.css('background-image', 'url(' + e.target.result + ')');
                imagePreview.hide().fadeIn(450);
                imageSection.stop(true, true).slideDown(350);
                dropzone.addClass('is-active');
            };

            reader.readAsDataURL(selectedFile);
        }
    }

    $('#imageUpload').change(function () {
        result.hide().text('');
        readURL(this);
    });

    $('#btn-predict').click(function () {
        if (!$('#imageUpload').val()) {
            showResult('Please choose an image before prediction.', true);
            return;
        }

        var formData = new FormData($('#upload-file')[0]);

        predictButton.prop('disabled', true).text('Predicting...');
        loader.stop(true, true).fadeIn(200);
        result.hide();

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: formData,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                showResult(data, false);
                console.log('Success!');
            },
            error: function (xhr) {
                var errorMessage = 'Prediction failed. Please try another image.';
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    errorMessage = xhr.responseJSON.error;
                }
                showResult(errorMessage, true);
            },
            complete: function () {
                loader.fadeOut(200);
                predictButton.prop('disabled', false).text('Predict Disease');
            }
        });
    });
});
