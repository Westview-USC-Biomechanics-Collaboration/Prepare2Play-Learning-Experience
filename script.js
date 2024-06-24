let sport; 
let movement;
let vid;
const progressBar = document.getElementsByClassName("progress-bar")[0];

function selectedVideo(self) {
    var file = self.files[0];
    var reader = new FileReader();

    reader.onload = function(e) {
        var src = e.target.result;
        var video = document.getElementById("video");
        var source = document.getElementById("source");
        
        source.setAttribute("src", src);
        
        video.load();
        //video.play();

        vid = video;
    }

    reader.readAsDataURL(file);
}

document.addEventListener('DOMContentLoaded', function() {
    var showButton = document.getElementById('showContainerButton');
    var hiddenContainer = document.getElementById('hiddenContainer');
    var sportSelect = document.getElementById('sport');
    var sportContainer = document.getElementById('sportContainer');
    var videoContainer = document.getElementById('videoContainer');
    var processButton = document.getElementById('finishButton');
    var informationContainer= document.getElementById('informationContainer');
    var text = document.getElementById("text");

    showButton.addEventListener('click', function() {
        sport = sportSelect.value; 
        hiddenContainer.classList.remove('hidden');

        if (!hiddenContainer.classList.contains('hidden')) {
            var finishButton = document.getElementById('finish');
            var movementSelect = document.getElementById('movement');

            finishButton.addEventListener('click', function() {
                movement = movementSelect.value; 
                console.log(movement); 
            });
        }
    });

    processButton.addEventListener('click', function() {
        if (!sport || sport === "") {
            sportContainer.style.backgroundColor = 'lightcoral';
        } else {
            sportContainer.style.backgroundColor = 'white';
        }
        
        if (!movement || movement === "") {
            hiddenContainer.style.backgroundColor = 'lightcoral';
        } else {
            hiddenContainer.style.backgroundColor = 'white';
        }

        if (!vid) {
            videoContainer.style.backgroundColor = 'lightcoral';
        } else {
            videoContainer.style.backgroundColor = 'white';
        }

        if (sport && movement && vid) {
            progressBar.style.display = "block";
            progressBar.style.setProperty('--width', 10); 

            let width = 0;
            const interval = setInterval(() => {
                width += 0.1;
                progressBar.style.setProperty('--width', width);

                console.log(progressBar.style.getPropertyValue('--width')); 

                if (width >= 80) {
                    clearInterval(interval);
                    progressBar.style.display = "none"; 
                    progressBar.style.setProperty('--width', 1); 
                    informationContainer.style.display = "block";
                    text.textContent = `Sport: ${sport}, Movement: ${movement}` + "put graph here";
                   
                }

                
            }, 5);


            

        }
    });
});
