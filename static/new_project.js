// Array to store all uploaded files
let allUploadedFiles = [];
const imageUpload = document.getElementById('imageUpload');
const imageContainer = document.getElementById('imageContainer');
const actionButtons = document.getElementById('actionButtons');
const uploadContainer = document.querySelector('.upload-container');
const imageGrid = document.getElementById('imageGrid');
const generateMask = document.getElementById('generateMask');
const message = document.getElementById("message");
const removeAllButton = document.getElementById('removeAll');
const addMoreButton = document.getElementById('addMoreImages');
const loadingScreen = document.getElementById("loadingScreen");
const rhsActionButtons = document.getElementById("rhsActionButtons");
const maskContainer = document.getElementById("maskContainer");
const saveMask = document.getElementById("saveMask");
const dialogueBox = document.getElementById("dialogueBox");
const projectNameInput = document.getElementById("projectNameInput");
const submitProjectName = document.getElementById("submitProjectName");

imageUpload.addEventListener('change', function () {
    fetch("/remove_temporary", {
        method: "POST",
    })
        .then(response => response.json())
        .catch((error) => {
            console.error("Error:", error);
            alert("Failed to remove temporary files.");
        });

    const files = Array.from(imageUpload.files);
    if (files.length > 0) {
        allUploadedFiles = [...files]; // Initialize with the initial upload files
        imageContainer.style.display = 'flex'; // Show image container
        actionButtons.style.display = 'flex'; // Show action buttons
        imageGrid.innerHTML = ''; // Clear previous images

        files.forEach(file => {
            const reader = new FileReader();
            reader.onload = function (e) {
                const imgDiv = document.createElement('div');
                imgDiv.className = 'col-2';
                imgDiv.innerHTML = `<img src="${e.target.result}" class="img-fluid" alt="Uploaded Image">`;
                imageGrid.appendChild(imgDiv);
            };
            reader.readAsDataURL(file);
        });

        uploadContainer.style.display = 'none'; // Hide upload button
    }

    message.innerHTML = `Click "generate mask" to continue.`; // Clear previous message
});

addMoreButton.addEventListener('click', function () {
    // Create a temporary input element to allow additional file uploads
    const tempInput = document.createElement('input');
    tempInput.type = 'file';
    tempInput.accept = 'image/*';
    tempInput.multiple = true; // Allow multiple file uploads   
    tempInput.click(); // Trigger the file picker dialog

    // Add an event listener to handle the files once the user selects them
    tempInput.addEventListener('change', function () {
        const newFiles = Array.from(tempInput.files); // Get the newly uploaded files
        newFiles.forEach(file => {
            allUploadedFiles.push(file); // Add to the global array
            const reader = new FileReader();
            reader.onload = function (e) {
                const imgDiv = document.createElement('div');
                imgDiv.className = 'col-2';
                imgDiv.innerHTML = `<img src="${e.target.result}" class="img-fluid" alt="Uploaded Image">`;
                imageGrid.appendChild(imgDiv);
            };
            reader.readAsDataURL(file);
        });
    });
});

removeAllButton.addEventListener('click', function () {
    imageGrid.innerHTML = ''; // Clear all images
    imageContainer.style.display = 'none'; // Hide image container
    maskContainer.style.display = 'none'; // Hide image container
    actionButtons.style.display = 'none'; // Hide action buttons
    rhsActionButtons.style.display = 'none'; // Hide action buttons
    uploadContainer.style.display = 'block'; // Show upload button again

    message.style.display = 'block'; // show
    message.innerHTML = `Please upload at least one image to continue.`; // Clear previous message
    loadingScreen.style.display = "none";
    allUploadedFiles = []; // Reset the global array

    // Send signal to Flask backend
    fetch("/remove_temporary", {
        method: "POST",
    })
        .then((response) => response.json())
        .catch((error) => {
            console.error("Error:", error);
            alert("Failed to remove temporary files.");
        });
});


generateMask.addEventListener('click', function () {
    const message = document.getElementById("message");
    message.innerHTML = "<h4>Generating Mask(s)</h4>";
    loadingScreen.style.display = "block";

    const imageInput = document.getElementById("imageUpload");
    const uploadedFiles = imageInput.files; // Get all uploaded files

    // Disable "Add More" and "Remove All" buttons
    addMoreButton.disabled = true;
    removeAllButton.disabled = true;

    if (uploadedFiles.length === 0) {
        alert("Please upload at least one image before generating a mask.");
        loadingScreen.style.display = "none";
        return;
    }

    // Create FormData to send files
    // Create FormData with all uploaded files
    const formData = new FormData();
    allUploadedFiles.forEach(file => {
        formData.append("files", file);
    });

    // Step 1: Upload images
    fetch("/upload_images", {
        method: "POST",
        body: formData,
    })
        .then((response) => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json(); // Only parse if response is OK
        })
        .then((data) => {
            if (data.success) {
                return fetch("/generate_masks", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ saved_files: data.saved_files }),
                });
            } else {
                throw new Error("File upload failed: " + data.error);
            }
        })
        .then((response) => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json(); // Only parse if response is OK
        })
        .then((data) => {
            if (data.success) {
                loadingScreen.style.display = "none";
                message.style.display = "none";

                maskContainer.style.display = "block";
                rhsActionButtons.style.display = "block";

                const maskGrid = document.getElementById("maskGrid");
                maskGrid.innerHTML = ''; // Clear previous masks
                data.mask_paths.forEach((maskFilename) => {
                    maskFilename = maskFilename.replace(/\\/g, '/');
                    const maskDiv = document.createElement('div');
                    maskDiv.className = 'col-2';
                    maskDiv.innerHTML = `<img src="${maskFilename}" class="img-fluid" alt="Generated Mask">`;
                    maskGrid.appendChild(maskDiv);
                });

                addMoreButton.disabled = false;
                removeAllButton.disabled = false;
                generateMask.innerHTML = "Regenerated Mask";

            } else {
                throw new Error("Error generating masks: " + data.error);
            }
        })
        .catch((error) => {
            console.error("Error:", error);
            alert("Something went wrong: " + error.message);
        });

});

// Show the dialog box when saveMask is clicked
saveMask.addEventListener("click", function () {
    projectNameInput.value = ""; // Clear previous input
    dialogueBox.style.display = "block"; // Show dialog box
});

// Handle submission of the project name
submitProjectName.addEventListener("click", function () {
    const projectName = projectNameInput.value.trim();

    if (!projectName) {
        alert("Project name is required to save the mask.");
        return;
    }

    // Send the project name to the server
    fetch("/save_mask", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ project_name: projectName }),
    })
        .then((response) => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then((data) => {
            if (data.success) {
                alert("Mask saved successfully.");

                window.location.href = `/projects/${encodeURIComponent(projectName)}`;
            } else {
                throw new Error(data.error || "Unknown error occurred.");
            }
        })
        .catch((error) => {
            console.error("Error:", error);
            alert("Failed to save mask. Please try again.");
        });

    dialogueBox.style.display = "none"; // Hide dialog box after submission
});

// Optional: Close dialog box on outside click
window.addEventListener("click", function (event) {
    if (event.target === dialogueBox) {
        dialogueBox.style.display = "none";
    }
});
