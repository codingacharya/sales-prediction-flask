// script.js - form validation, AJAX submit to /api/predict, result rendering

document.addEventListener('DOMContentLoaded', () => {
  const form = document.querySelector('#predict-form');
  const inputs = Array.from(document.querySelectorAll('.predict-input'));
  const submitBtn = document.querySelector('#predict-btn');
  const resultCard = document.querySelector('#result-card');
  const resultValue = document.querySelector('#result-value');
  const resultNote = document.querySelector('#result-note');
  const errorBox = document.querySelector('#error-box');

  // simple numeric validation
  function validate() {
    errorBox.style.display = 'none';
    const values = inputs.map(i => i.value.trim());
    for (let v of values) {
      if (v === '' || isNaN(v)) {
        return { ok: false, message: 'Please enter valid numeric values for all fields.' };
      }
    }
    return { ok: true, values: values.map(v => parseFloat(v)) };
  }

  // show error
  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.style.display = 'block';
    resultCard.style.display = 'none';
  }

  // show result
  function showResult(value, inputs) {
    resultValue.textContent = value;
    resultNote.textContent = `Input: ${inputs.join(', ')}`;
    resultCard.style.display = 'block';
    errorBox.style.display = 'none';
  }

  // handle AJAX submission
  async function submitAjax(payload) {
    submitBtn.disabled = true;
    submitBtn.textContent = 'Predicting...';
    try {
      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        const errorData = await resp.json().catch(() => ({}));
        throw new Error(errorData.error || `Server returned ${resp.status}`);
      }

      const data = await resp.json();
      if (data.predicted_sales !== undefined) {
        showResult(data.predicted_sales, Object.values(payload));
      } else {
        throw new Error('Invalid response from server.');
      }
    } catch (err) {
      showError(err.message || 'Prediction failed.');
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = 'Predict';
    }
  }

  // on submit
  if (form) {
    form.addEventListener('submit', (e) => {
      // prevent default to use AJAX; if AJAX fails, user can still rely on server-side form action if you prefer
      e.preventDefault();
      const valid = validate();
      if (!valid.ok) {
        showError(valid.message);
        return;
      }

      // build payload using input name attributes
      const payload = {};
      inputs.forEach((input, idx) => {
        payload[input.name] = parseFloat(input.value.trim());
      });

      submitAjax(payload);
    });
  }
});
