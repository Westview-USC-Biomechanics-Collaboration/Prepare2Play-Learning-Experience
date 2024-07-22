function scrollToDiv(divId) {
    const element = document.getElementById(divId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
    }
}

function selectMovement(movement) {
    localStorage.setItem('selectedMovement', movement);
    window.location.href = '/SportsTemplate';
}

function updateContent() {
    const movementTitle = localStorage.getItem('selectedMovement') || 'Unknown Movement';
    document.getElementById('sportTitle').textContent = movementTitle;
}

document.addEventListener('DOMContentLoaded', function () {
    if (window.location.pathname === '/index') {
        const movementLinks = document.querySelectorAll('.nested-dropdown a');
        movementLinks.forEach(link => {
            link.addEventListener('click', function (event) {
                event.preventDefault();
                const movement = this.textContent;
                selectMovement(movement);
            });
        });
    }

    const accordions = document.querySelectorAll('.accordion li label');
    accordions.forEach(label => {
        label.addEventListener('click', function () {
            const content = this.nextElementSibling;
            const expanded = this.getAttribute('aria-expanded') === 'true';

            if (expanded) {
                content.style.maxHeight = null;
                content.style.padding = '0 10px';
                this.setAttribute('aria-expanded', 'false');
                this.querySelector('span').textContent = '+';
            } else {
                content.style.maxHeight = content.scrollHeight + 'px';
                content.style.padding = '10px 10px 20px';
                this.setAttribute('aria-expanded', 'true');
                this.querySelector('span').textContent = '-';
            }
        });
    });

    if (window.location.pathname === '/SportsTemplate') {
        updateContent();

        const videoFileInput = document.getElementById('videoFile');
        const uploadedVideo = document.getElementById('uploadedVideo');
        const videoContainer = document.getElementById('videoContainer');

        if (videoFileInput && uploadedVideo) {
            videoFileInput.addEventListener('change', function () {
                const file = this.files[0];
                if (file) {
                    const fileURL = URL.createObjectURL(file);
                    uploadedVideo.src = fileURL;
                    videoContainer.style.display = 'block';
                    uploadAndProcessVideo(file);
                }
            });
        }
    }
});

function uploadAndProcessVideo(videoFile) {
    const formData = new FormData();
    formData.append('video', videoFile);

    fetch('/process_video', {
        method: 'POST',
        body: formData
    }).then(response => response.json())
    .then(data => {
        console.log('Video uploaded successfully:', data);
    }).catch(error => {
        console.error('Error uploading video:', error);
    });
}
