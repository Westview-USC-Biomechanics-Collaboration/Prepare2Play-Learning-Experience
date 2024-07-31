function scrollToDiv(divId) {
      var element = document.getElementById(divId);
      element.scrollIntoView({ behavior: 'smooth' });
  }
  
  function selectMovement(movement) {
    localStorage.setItem('selectedMovement', movement);
  }

  document.addEventListener('DOMContentLoaded', function () {
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
});

function debugging() {
    movement = localStorage.getItem('selectedMovement');
    console.log(movement);
}

document.addEventListener('DOMContentLoaded', function () {
    console.log("Document is ready");

    var modal = document.getElementById("about-us-modal");
    var button = document.getElementById("about-us-button");
    var closeButton = document.getElementsByClassName("close-button")[0];

    if (modal) {
        console.log("Modal found:", modal);
    } else {
        console.log("Modal not found");
    }

    if (button) {
        console.log("Button found:", button);
    } else {
        console.log("Button not found");
    }

    if (closeButton) {
        console.log("Close button found:", closeButton);
    } else {
        console.log("Close button not found");
    }

    if (button) {
        button.onclick = function (event) {
            console.log("Button clicked");
            if (modal) {
                modal.style.display = "block";
                console.log("Modal display set to block");
            }
        };
    }

    if (closeButton) {
        closeButton.onclick = function () {
            console.log("Close button clicked");
            if (modal) {
                modal.style.display = "none";
                console.log("Modal display set to none");
            }
        };
    }

    window.onclick = function (event) {
        if (event.target == modal) {
            console.log("Outside modal clicked");
            if (modal) {
                modal.style.display = "none";
                console.log("Modal display set to none");
            }
        }
    };
});

function selectMovement(movement) {
    localStorage.setItem('selectedMovement', movement);
    console.log(localStorage.getItem('selectedMovement'));
    window.location.href = 'SportsTemplate.html'; // Redirect to SportsTemplate.html
}

const videoFileInput = document.getElementById('videoFile');
const uploadedVideo = document.getElementById('uploadedVideo');

document.getElementById('videoContainer').style.display = 'block';

videoFileInput.addEventListener('change', function() {
    const file = this.files[0];
    

    const fileURL = URL.createObjectURL(file);
    uploadedVideo.src = fileURL;
    

    uploadAndProcessVideo(file);
});

function uploadAndProcessVideo(videoFile) {

    fetch('/process_video', {
        method: 'POST',
        body: videoFile
    }).then(response => {

    }).catch(error => {
        console.error('Error uploading video:', error);
    });
}

window.onload = debugging;

