$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    $('#analysis').hide();

    var uploadedDataUrl = null;

    // Escape text coming from the API before inserting it as HTML
    function esc(value) {
        return $('<div>').text(value === null || value === undefined ? '' : String(value)).html();
    }

    function fillList(selector, items) {
        var list = $(selector).empty();
        (items || []).forEach(function (item) {
            list.append($('<li>').text(item));
        });
        return list;
    }

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                uploadedDataUrl = e.target.result;
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
        $('#result').html('');
        $('#result').hide();
        $('#analysis').hide();
        readURL(this);
    });

    // Original summary box (kept from the minor project)
    function renderSummary(data) {
        var top3Html = '';
        if (data.top3_crops && data.top3_crops.length > 0) {
            top3Html = '<div class="result-subtitle">Top 3 Crop Predictions</div><ol class="top3-list">';
            data.top3_crops.forEach(function (item) {
                top3Html += '<li><span class="crop-name">' + esc(item.crop) + '</span><span class="crop-confidence">' + esc(item.confidence) + '%</span></li>';
            });
            top3Html += '</ol>';
        }

        var resultHtml =
            '<div class="result-title">Prediction Result</div>' +
            '<div class="result-line"><strong>Crop:</strong> ' + esc(data.predicted_crop) + '</div>' +
            '<div class="result-line"><strong>Disease:</strong> ' + esc(data.predicted_disease) + '</div>' +
            '<div class="result-line"><strong>Confidence:</strong> ' + esc(data.confidence) + '%</div>' +
            top3Html;

        $('#result').fadeIn(600);
        $('#result').html(resultHtml);
    }

    function renderWarnings(warnings) {
        var box = $('#analysis-warnings').empty();
        (warnings || []).forEach(function (w) {
            box.append($('<div class="alert alert-warning analysis-alert" role="alert">').text(w));
        });
    }

    function renderPrediction(data) {
        var p = data.prediction || {};
        $('#analysis-uploaded').attr('src', uploadedDataUrl || '');
        $('#pred-crop').text(p.crop || data.predicted_crop);
        $('#pred-disease').text(p.disease || data.predicted_disease);
        $('#pred-confidence').text((p.confidence !== undefined ? p.confidence : data.confidence) + '%');
        var conf = Math.max(0, Math.min(100, Number(p.confidence || data.confidence || 0)));
        $('#pred-confidence-bar').css('width', conf + '%').toggleClass('low', !!p.low_confidence);
        if (p.low_confidence) {
            $('#pred-confidence-note').text('Below the ' + p.low_confidence_threshold + '% confidence threshold: treat this identification as uncertain.').addClass('warn');
        } else {
            $('#pred-confidence-note').text('Confidence threshold for reliable identification: ' + (p.low_confidence_threshold || 60) + '%.').removeClass('warn');
        }
        var list = $('#pred-top3').empty();
        (p.top3_diseases || []).forEach(function (d) {
            list.append($('<li>').append(
                $('<span>').text(d.crop + ' - ' + d.disease),
                $('<strong>').text(d.confidence + '%')
            ));
        });
    }

    function renderSeverity(sev) {
        sev = sev || {};
        var category = sev.category || 'Unknown';
        var cls = category.toLowerCase().replace(/\s+/g, '-');
        $('#sev-badge').attr('class', 'sev-badge ' + cls).text(category);
        $('#sev-percentage').text(sev.percentage !== null && sev.percentage !== undefined
            ? sev.percentage.toFixed(2) + '% of leaf area affected' : 'No percentage reported');
        if (sev.percentage !== null && sev.percentage !== undefined) {
            var pos = Math.min(sev.percentage, 50) / 50 * 100;  // scale shows 0-50%
            $('#sev-marker').css('left', pos + '%').show();
            $('#sev-scale').show();
        } else {
            $('#sev-marker').hide();
        }
        var statusText = {
            estimated: 'Estimated (experimental)',
            unreliable: 'Unreliable - leaf segmentation failed quality checks',
            not_applicable: 'Not applicable for this disease',
            skipped: 'Skipped (healthy leaf)',
            error: 'Error during estimation'
        }[sev.status] || sev.status;
        $('#sev-status').text(statusText);
        $('#sev-method').text(sev.method ? 'HSV lesion-area ratio' + (sev.validity === 'low' ? ' (low-validity profile)' : '') : '-');
        $('#sev-note').text(sev.note || '');
        fillList('#sev-warnings', sev.warnings);
    }

    function renderVisualization(viz) {
        if (!viz) {
            $('#viz-card').hide();
            return;
        }
        $('#viz-card').show();
        var views = {original: '#viz-original', leaf_mask: '#viz-leaf', lesion_mask: '#viz-lesion', overlay: '#viz-overlay'};
        Object.keys(views).forEach(function (key) {
            var img = $(views[key]);
            if (viz[key]) {
                img.attr('src', viz[key]).closest('figure').show();
            } else {
                img.removeAttr('src').closest('figure').hide();
            }
        });
    }

    function renderChemical(chem) {
        var box = $('#treat-chemical').empty();
        if (!chem) {
            box.text('No chemical guidance available.');
            return;
        }
        box.attr('class', 'chemical-box ' + chem.status);
        box.append($('<p class="chem-summary">').text(chem.summary));
        if (chem.generic_guidance && chem.generic_guidance.length) {
            var ul = $('<ul class="plain-list bullet">');
            chem.generic_guidance.forEach(function (g) { ul.append($('<li>').text(g)); });
            box.append(ul);
        }
        if (chem.active_ingredients_named_by_source && chem.active_ingredients_named_by_source.length) {
            var details = $('<details>');
            details.append($('<summary>').text('Active ingredients named by the cited sources (' + chem.active_ingredients_named_by_source.length + ') - requires expert review'));
            details.append($('<p class="small-note">').text(chem.active_ingredients_named_by_source.join(', ')));
            box.append(details);
        }
        (chem.notes || []).forEach(function (n) { box.append($('<p class="small-note">').text(n)); });
        if (chem.status !== 'not_indicated' && chem.status !== 'suppressed_low_confidence') {
            box.append($('<p class="small-note label-note">').text(chem.label_note));
        }
    }

    function renderTreatment(t) {
        t = t || {};
        var info = $('#treat-disease-info').empty();
        info.append($('<h3 class="sub-title">').text(t.crop + ' - ' + t.disease));
        if (t.disease_info) {
            var di = t.disease_info;
            var meta = [];
            if (di.pathogen) meta.push('Cause: ' + di.pathogen);
            if (di.pathogen_type) meta.push('Type: ' + di.pathogen_type.replace('_', ' '));
            if (di.vector) meta.push('Vector: ' + di.vector);
            info.append($('<p class="small-note">').text(meta.join(' | ')));
            if (di.tentative) {
                info.append($('<p class="small-note warn">').text('Tentative identification (low confidence).'));
            }
            if (di.symptoms && di.symptoms.length) {
                var sym = $('<ul class="plain-list bullet">');
                di.symptoms.forEach(function (s) { sym.append($('<li>').text(s)); });
                info.append($('<div class="small-note">').text('Typical symptoms:'), sym);
            }
        }
        if (t.expert_consultation && t.expert_consultation.recommended) {
            $('#treat-expert').text('Expert consultation recommended: ' + t.expert_consultation.reason).show();
        } else {
            $('#treat-expert').hide();
        }
        fillList('#treat-management', t.management_guidance);
        $('#treat-management-block').toggle(!!(t.management_guidance && t.management_guidance.length));
        fillList('#treat-vector', t.vector_management);
        $('#treat-vector-block').toggle(!!(t.vector_management && t.vector_management.length));
        fillList('#treat-cultural', (t.cultural_practices || []).length ? t.cultural_practices : ['-']);
        fillList('#treat-preventive', (t.preventive_measures || []).length ? t.preventive_measures : ['-']);
        fillList('#treat-monitoring', t.monitoring);
        renderChemical(t.chemical_guidance);

        var src = $('#treat-sources').empty();
        (t.sources || []).forEach(function (s) {
            src.append($('<li>').append(
                $('<a target="_blank" rel="noopener noreferrer">').attr('href', s.url).text(s.title),
                $('<span class="small-note">').text(' - ' + s.organization)
            ));
        });
        if (!t.sources || !t.sources.length) {
            src.append($('<li class="small-note">').text('No verified source for this entry.'));
        }
        $('#treat-disclaimer').text((t.review_status ? 'Knowledge-base status: ' + t.review_status + '. ' : '') + (t.disclaimer || ''));
    }

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
        $('#analysis').hide();

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
                renderSummary(data);
                if (data.prediction) {
                    renderWarnings(data.warnings);
                    renderPrediction(data);
                    renderSeverity(data.severity);
                    renderVisualization(data.visualization);
                    renderTreatment(data.treatment);
                    $('#analysis').fadeIn(600);
                }
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
