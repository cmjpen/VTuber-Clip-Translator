document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("translateButton").addEventListener("click", submitVideo);
});

async function submitVideo() {
    const youtubeLink = document.getElementById("youtubeLink").value;
    console.log("YouTube Link:", youtubeLink);
    const statusElement = document.getElementById("status");

    if (!youtubeLink) {
        statusElement.innerText = "Please enter a YouTube link.";
        return;
    }
    
    // Disable the button to prevent multiple submissions
    translateButton.disabled = true;
    
    // Extract the video ID from the YouTube link
    const videoId = youtubeLink.split('v=')[1];
    const ampersandPosition = videoId.indexOf('&');
    if (ampersandPosition !== -1) {
        videoId = videoId.substring(0, ampersandPosition);
    }

    // Show the video container
    document.querySelector('.video-container').classList.remove('hidden');

    // Update the iframe src attribute with the new video ID
    const youtubeVideo = document.getElementById("youtubeVideo");
    youtubeVideo.src = `https://www.youtube.com/embed/${videoId}`;

    statusElement.innerText = "Processing...";

    try {
        const response = await fetch("http://localhost:8080/translate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: youtubeLink })
        });

        const data = await response.json();
        if (response.ok) {
            statusElement.innerText = data.message || "Translation complete!";
        } else {
            statusElement.innerText = "Error: " + (data.error || "Unknown error");
        }
    } catch (error) {
        statusElement.innerText = "Failed to connect to the server.";
    } finally {
        // Re-enable the button after processing is complete
        translateButton.disabled = false;
    }
}
