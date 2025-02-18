function displayFileName() {
    const fileInput = document.getElementById("pdfFile");
    const fileNameDisplay = document.getElementById("file-name");
    
    if (fileInput.files.length > 0) {
        fileNameDisplay.textContent = `Selected: ${fileInput.files[0].name}`;
        fileNameDisplay.style.color = "#ffffff"; // Ensure visibility
    } else {
        fileNameDisplay.textContent = "No file chosen";
    }
}

function uploadPDF() {
    let fileInput = document.getElementById("pdfFile");
    let uploadButton = document.getElementById("uploadButton");
    let loadingDiv = document.getElementById("loading");

    if (fileInput.files.length === 0) {
        alert("Please select a file first!");
        return;
    }

    let formData = new FormData();
    formData.append("file", fileInput.files[0]);

    // Hide upload button and show loader
    uploadButton.style.display = "none";
    loadingDiv.style.display = "block";  // Ensure loader is visible

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            window.location.href = data.redirect; // Redirect to chat page
        } else {
            alert("File upload failed.");
            uploadButton.style.display = "block";  // Show button again
            loadingDiv.style.display = "none";   // Hide loader
        }
    })
    .catch(error => {
        alert("Error uploading file.");
        uploadButton.style.display = "block";
        loadingDiv.style.display = "none";
    });
}
