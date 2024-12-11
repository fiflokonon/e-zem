
    let isRecording = false;
    let mediaRecorder;
    let audioChunks = [];
    const statusElement = document.getElementById("status");
    const startRecordingButton = document.getElementById("start-recording");
    const audioPlayback = document.getElementById("audio-playback");
    const audioSource = document.getElementById("audio-source");

    // Demander l'accès au microphone
    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);

            // Commencer l'enregistrement
            mediaRecorder.onstart = () => {
                audioChunks = [];
                //statusElement.textContent = "Enregistrement en cours...";
                statusElement.textContent = "...";
                statusElement.classList.remove("stopped");
                statusElement.classList.add("recording");
            };

            // Ajouter des morceaux d'audio pendant l'enregistrement
            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };

            // Lorsque l'enregistrement est arrêté, on crée un fichier audio
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                audioSource.src = audioUrl;
                audioPlayback.style.display = 'block';
                //statusElement.textContent = "Enregistrement terminé. Écoutez-le ci-dessus.";
                statusElement.textContent = "-";
                statusElement.classList.remove("recording");
                statusElement.classList.add("stopped");
            };

            // Démarrer l'enregistrement
            mediaRecorder.start();
        } catch (error) {
            statusElement.textContent = "Erreur d'accès au microphone.";
            statusElement.classList.add("error");
            console.error("Erreur d'enregistrement:", error);
        }
    }

    // Arrêter l'enregistrement
    function stopRecording() {
        if (mediaRecorder) {
            mediaRecorder.stop();
        }
    }

    // Toggle entre démarrer et arrêter l'enregistrement
    startRecordingButton.addEventListener("click", () => {
        if (isRecording) {
            stopRecording();
            startRecordingButton.innerHTML = '<i class="fas fa-microphone"></i><span>Commande vocale</span>';
        } else {
            startRecording();
            startRecordingButton.innerHTML = '<i class="fas fa-stop"></i><span>Arrêter l\'enregistrement</span>';
        }
        isRecording = !isRecording;
    });
