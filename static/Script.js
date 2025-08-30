document.getElementById('imageInput').addEventListener('change', function (event) {
    const image = document.getElementById('uploadedImage');
    const file = event.target.files[0];

    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            image.src = e.target.result;
            image.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById('generateCaptionButton').addEventListener('click', async function () {
    const imageInput = document.getElementById('imageInput');
    const captionText = document.getElementById('captionText');
    const modelSelect = document.getElementById('modelSelect');

    if (!imageInput.files[0]) {
        captionText.textContent = "Please upload an image first.";
        return;
    }

    const formData = new FormData();
    formData.append('image', imageInput.files[0]);
    formData.append('model', modelSelect.value);

    captionText.textContent = "Generating caption...";

    try {
        const response = await fetch('/generate_caption', {
            method: 'POST',
            body: formData,
        });

        if (response.ok) {
            const data = await response.json();
            captionText.textContent = data.caption;
        } else {
            captionText.textContent = "Error generating caption.";
        }
    } catch (error) {
        captionText.textContent = error.message;
    }
});