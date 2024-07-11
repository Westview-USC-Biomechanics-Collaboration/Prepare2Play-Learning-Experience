// Function to scroll to a specific div
function scrollToDiv(divId) {
    var element = document.getElementById(divId);
    element.scrollIntoView({ behavior: 'smooth' });
}

// Function to select a movement and store it in localStorage
function selectMovement(movement) {
    localStorage.setItem('selectedMovement', movement);
    window.location.href = 'SportsTemplate.html'; // Redirect to SportsTemplate.html
}

// Function to update the page content on SportsTemplate.html
function updateContent() {
    const movementTitle = localStorage.getItem('selectedMovement') || 'Unknown Movement';
    document.getElementById('sportTitle').textContent = movementTitle;
}

// Add event listeners for the accordion functionality and movement selection
document.addEventListener('DOMContentLoaded', function () {
    const movementLinks = document.querySelectorAll('.nested-dropdown a');
    movementLinks.forEach(link => {
        link.addEventListener('click', function (event) {
            event.preventDefault();
            const movement = this.textContent;
            selectMovement(movement);
        });
    });

    // Accordion functionality
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

    // Call updateContent() only if on the SportsTemplate.html page
    if (document.getElementById('sportTitle')) {
        updateContent();
    }

    // Video file upload functionality
    const videoFileInput = document.getElementById('videoFile');
    const uploadedVideo = document.getElementById('uploadedVideo');

    if (videoFileInput && uploadedVideo) {
        document.getElementById('videoContainer').style.display = 'block';

        videoFileInput.addEventListener('change', function() {
            const file = this.files[0];
            const fileURL = URL.createObjectURL(file);
            uploadedVideo.src = fileURL;
            uploadAndProcessVideo(file);
        });
    }
});

// Function to handle video upload and processing
function uploadAndProcessVideo(videoFile) {
    fetch('/process_video', {
        method: 'POST',
        body: videoFile
    }).then(response => {
        // Handle the response
    }).catch(error => {
        console.error('Error uploading video:', error);
    });
}
