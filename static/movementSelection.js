function updateStatsVisibility(movement) {
    const statsMapping = {
    'Chest Pass': ['vertImpulse-Stat', 'horizImpulse-Stat', 'launchVel-Stat', 'launchAngle-Stat'],
    'Free Throw': ['vertImpulse-Stat', 'horizImpulse-Stat', 'launchVel-Stat', 'launchAngle-Stat', 'makesBasket-Stat'],
    'Jump Shot': ['vertImpulse-Stat', 'horizImpulse-Stat', 'launchVel-Stat', 'launchAngle-Stat', 'makesBasket-Stat'],
    'Football Pass': ['vertImpulse-Stat', 'horizImpulse-Stat', 'launchVel-Stat', 'launchAngle-Stat'],
    'Corner Kick': ['vertImpulse-Stat', 'horizImpulse-Stat'],
    'Offensive Header': ['vertImpulse-Stat', 'horizImpulse-Stat', 'launchVel-Stat', 'launchAngle-Stat'],
    'Softball Swing': ['vertImpulse-Stat', 'horizImpulse-Stat', 'launchVel-Stat', 'launchAngle-Stat'],
    'Backhand Slice': ['vertImpulse-Stat', 'horizImpulse-Stat', 'launchVel-Stat', 'launchAngle-Stat'],
    'Forehand': ['vertImpulse-Stat', 'horizImpulse-Stat', 'launchVel-Stat', 'launchAngle-Stat'],
    'Volley': ['vertImpulse-Stat', 'horizImpulse-Stat', 'launchVel-Stat', 'launchAngle-Stat'],
    'Axel Jump': ['vertImpulse-Stat', 'horizImpulse-Stat'],
    'Side Kick': ['vertImpulse-Stat', 'horizImpulse-Stat'],
    'Sprint Start': ['vertImpulse-Stat', 'horizImpulse-Stat'],
    'Volleyball Block': ['vertImpulse-Stat', 'horizImpulse-Stat'],
    'Volleyball Serve': ['vertImpulse-Stat', 'horizImpulse-Stat', 'launchVel-Stat', 'launchAngle-Stat'],
    'Volleyball Set': ['vertImpulse-Stat', 'horizImpulse-Stat'],
    'Change in Direction in Tag': ['vertImpulse-Stat', 'horizImpulse-Stat'],
    'Frisbee Pull': ['vertImpulse-Stat', 'horizImpulse-Stat'],
    'Frisbee Throw': ['vertImpulse-Stat', 'horizImpulse-Stat', 'launchVel-Stat', 'launchAngle-Stat'],
    'Hopscotch': ['vertImpulse-Stat', 'horizImpulse-Stat'],
    'Jump Rope': ['vertImpulse-Stat', 'horizImpulse-Stat'],
    'Yoga Chair Pose': ['vertImpulse-Stat', 'horizImpulse-Stat'],
    'Surfing': ['vertImpulse-Stat', 'horizImpulse-Stat']
    };

    console.log('updateStatsVisibility called with movement:', movement);

    document.querySelectorAll('.Results p').forEach(stat => {
        console.log('Hiding:', stat.id);
        stat.style.display = 'none';
    });

    const relevantStats = statsMapping[movement];
    console.log('Relevant stats:', relevantStats);
    if (relevantStats) {
        relevantStats.forEach(statId => {
            const element = document.getElementById(statId);
            if (element) {
                console.log('Showing:', statId);
                element.style.display = 'block';
            }
        });
    }
}

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
    console.log('updateContent() called');

    const movementTitle = localStorage.getItem('selectedMovement') || 'Unknown Movement';
    console.log(`Selected movement: ${movementTitle}`);
    document.getElementById('sportTitle').textContent = movementTitle;

    updateStatsVisibility(movementTitle);
}

console.log('CURRENT pathname in updateContent check:', window.location.pathname);


document.addEventListener('DOMContentLoaded', function () {
    if (window.location.pathname === '/index.html') {
        const movementLinks = document.querySelectorAll('.dropdown-content a');
        movementLinks.forEach(link => {
            link.addEventListener('click', function (event) {
                event.preventDefault();
                const movement = this.textContent.trim();
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

    if (window.location.pathname === '/SportsTemplate.html') {
        console.log('Calling updateContent()');

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
    fetch('/process_video', {
        method: 'POST',
        body: videoFile
    }).then(response => {
    }).catch(error => {
        console.error('Error uploading video:', error);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    const aboutUsLink = document.getElementById('about-us-link');
    const popup = document.getElementById('about-us-popup');
    const closeBtn = popup.querySelector('.close-btn');
    const iframe = document.getElementById('popup-iframe');

    aboutUsLink.addEventListener('click', (event) => {
        event.preventDefault();
        const url = aboutUsLink.getAttribute('href');
        iframe.src = url;
        popup.style.display = 'flex';
    });

    closeBtn.addEventListener('click', () => {
        popup.style.display = 'none';
        iframe.src = '';
    });

    window.addEventListener('click', (event) => {
        if (event.target === popup) {
            popup.style.display = 'none';
            iframe.src = '';
        }
    });
});

function debugging() {
    movement = localStorage.getItem('selectedMovement');
    console.log(movement);
}

const videoFileInput = document.getElementById('videoFile');
const uploadedVideo = document.getElementById('uploadedVideo');

//document.getElementById('videoContainer').style.display = 'block';

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