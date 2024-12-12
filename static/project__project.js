function openTab(tabId) {
    const tabs = document.querySelectorAll('.tab-content');
    const buttons = document.querySelectorAll('.tab-button');
    tabs.forEach(tab => tab.classList.remove('active'));
    buttons.forEach(button => button.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    event.currentTarget.classList.add('active');
}
function editDatabase() {
    alert('Edit Database button clicked!');
}
document.addEventListener("DOMContentLoaded", () => {
    const imageGrid = document.getElementById("imageGrid");
    const maskGrid = document.getElementById("maskGrid");

    const projectName = imageGrid.getAttribute("data-project-name");

    // Fetch images and masks for the project
    fetch(`/get_project_data/${projectName}`)
        .then(response => response.json())
        .then(data => {
            if (data.images) {
                imageGrid.innerHTML = ''; // Clear previous masks
                data.images.forEach((filename) => {
                    filename = filename.replace(/\\/g, '/');
                    filename = filename.split('/').pop();
                    const imageDiv = document.createElement('div');
                    imageDiv.className = 'col-2';
                    imageDiv.innerHTML = `<img src="/db/${projectName}/image/${encodeURIComponent(filename)}" class="img-fluid" alt="Generated Image">`;
                    imageGrid.appendChild(imageDiv);
                });

            }

            if (data.masks) {
                maskGrid.innerHTML = ''; // Clear previous masks
                data.images.forEach((filename) => {
                    filename = filename.replace(/\\/g, '/');
                    filename = filename.split('/').pop();
                    const imageDiv = document.createElement('div');
                    imageDiv.className = 'col-2';
                    imageDiv.innerHTML = `<img src="/db/${projectName}/masks/${encodeURIComponent(filename)}" class="img-fluid" alt="Generated Image">`;
                    maskGrid.appendChild(imageDiv);
                });
            }
        })
        .catch(error => console.error("Error loading project data:", error));
});


// Overlay View
function openOverlayViewTab(tabId) {
    const tabs = document.querySelectorAll('.tab-content');
    const buttons = document.querySelectorAll('.tab-button');
    tabs.forEach(tab => tab.classList.remove('active'));
    buttons.forEach(button => button.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    event.currentTarget.classList.add('active');

    const overlayGrid = document.getElementById("overlayGrid");
    const projectName = overlayGrid.getAttribute("data-project-name");

    // Fetch both images and masks from the server using the project name
    fetch(`/get_project_data/${projectName}`)
        .then(response => response.json())
        .then(data => {
            if (data.overlays) {
                overlayGrid.innerHTML = ''; // Clear previous masks
                data.overlays.forEach((filename) => {
                    filename = filename.replace(/\\/g, '/');
                    filename = filename.split('/').pop();
                    const imageDiv = document.createElement('div');
                    imageDiv.className = 'col-2';
                    imageDiv.innerHTML = `<img src="/db/${projectName}/overlay/${encodeURIComponent(filename)}" class="img-fluid" alt="Generated Image">`;
                    overlayGrid.appendChild(imageDiv);
                });
            }
        })
        .catch(error => console.error("Error loading project data:", error));
}


// ALERT SYSTEM
const imageUpload = document.getElementById("imageUpload");
const alertMessages = document.getElementById("alertMessages");
const alertList = document.querySelector(".alert-list");
const spinnerCnt = document.getElementById("spinnerCnt");

// Show alerts container after uploading images
imageUpload.addEventListener("change", (event) => {
    const files = Array.from(imageUpload.files);
    console.log(files);

    spinnerCnt.style.display = "block";

    const projectName = imageUpload.getAttribute("data-project-name");

    if (!projectName) {
        alert("Project name is missing.");
        return;
    }

    if (files.length === 0) {
        alert("No files selected.");
        return;
    }

    // Create FormData object to send files and project name
    const formData = new FormData();
    formData.append("project_name", projectName); // Append the project name

    // Ensure `files` is iterable, like a FileList or array
    if (files && files.length > 0) {
        Array.from(files).forEach((file, index) => {
            formData.append(`files[${index}]`, file); // Append each file using a consistent key
        });
    } else {
        console.error("No files provided or invalid files input.");
    }

    // Send files and project name to the server
    fetch("/clear_alert", {
        method: "POST",
        body: formData,
    })
        .then(response => response.json()) // Parse the response as JSON
        .then(data => {
            if (data.success) {
                // Handle successful alert clearing
                alertMessages.classList.add("hidden"); // Optionally hide alert messages
            } else {
                // Handle failure
                console.error("Failed to clear alerts:", data.message);
            }
        })
        .then(data => {
            fetch("/detect_changes", {
                method: "POST",
                body: formData,
            })
                .then(response => response.json()) // Parse the JSON response
                .then(data => {
                    if (data.n_files > 0) {
                        alertMessages.classList.remove("hidden");

                        // Handle TIRs (New Roads Detected)
                        data.TIRs.forEach((tir, index) => {
                            const alertItem = document.createElement("li");
                            alertItem.className = "alert-item";
                            if (data.IR_NEWs[index] >= data.IR_REMOVEDs[index]) {
                                alertItem.innerHTML = `
                                    <div class="alert-content">
                                        <h3>New Road Detected</h3>
                                        <small class="text-muted">Total Impact Ratio: ${tir}</small>
                                        <button class="dropdown-toggle">Details</button>
                                    </div>
                                    <div class="dropdown hidden">
                                        <img src="/db/${projectName}/alert/${data.FILEs[index]}" alt="Sample Image">
                                        <div class="btn-container">
                                            <button class="new-btn old-mask">Old Mask</button>
                                            <button class="new-btn new-mask">New Mask</button>
                                        </div>
                                    </div>
                                `;
                            }
                            else {
                                alertItem.innerHTML = `
                                    <div class="alert-content">
                                        <h3>Old Roads Removed: </h3>
                                        <small class="text-muted">Total Impact Ratio: ${tir}</small>
                                        <button class="dropdown-toggle">Details</button>
                                    </div>
                                    <div class="dropdown hidden">
                                        <img src="/db/${projectName}/alert/${data.FILEs[index]}" alt="Sample Image">
                                        <div class="btn-container">
                                            <button class="new-btn old-mask" id="old-mask">Keep Old Mask</button>
                                            <button class="new-btn new-mask" id="new-mask">Add New Mask</button>
                                        </div>
                                    </div>
                                `;
                            };

                            spinnerCnt.style.display = "none";

                            // Toggle dropdown visibility
                            alertItem.querySelector(".dropdown-toggle").addEventListener("click", () => {
                                const dropdown = alertItem.querySelector(".dropdown");
                                dropdown.classList.toggle("hidden");
                            });

                            alertList.appendChild(alertItem);
                        });
                    };
                })
                .catch(error => console.error("Error:", error));
        })
        .catch(error => {
            // Handle network or other errors
            console.error("Error clearing alerts:", error);
        });
});

const oldMask = document.getElementById("old-mask");
const newMask = document.getElementById("new-mask");

document.addEventListener("click", (event) => {
    // Check if the clicked element has the 'old-mask' class
    if (event.target.classList.contains("old-mask")) {
        const button = event.target;

        // Get the parent list item containing this button
        const alertItem = button.closest(".alert-item");

        if (!alertItem) {
            console.error("Could not find the parent alert item.");
            return;
        }

        // Get the file name from the image's src attribute
        const image = alertItem.querySelector("img");
        if (!image) {
            console.error("Could not find the image element.");
            return;
        }

        const fileSrc = image.getAttribute("src");
        const fileName = fileSrc.split("/").pop(); // Extract the file name from the path

        const projectName = imageUpload.getAttribute("data-project-name");
        if (!projectName) {
            console.error("Project name is missing.");
            alert("Project name is not available. Please try again.");
            return;
        }

        // Prepare the form data to send to the server
        const formData = new FormData();
        formData.append("project_name", projectName);
        formData.append("file_name", fileName);

        // Send the request to remove the specific file
        fetch("/clear_alerts", {
            method: "POST",
            body: formData,
        })
            .then((response) => {
                if (!response.ok) {
                    // Handle HTTP errors
                    throw new Error(`Server error: ${response.status}`);
                }
                return response.json(); // Parse the JSON response
            })
            .then((data) => {
                // Handle the server response
                if (data.success) {
                    console.log("File cleared successfully:", data.message);
                    alert("File has been cleared successfully.");

                    // Remove the specific list item from the alert list
                    alertItem.remove();
                } else {
                    console.warn("Failed to clear the file:", data.message);
                    alert("Failed to clear the file. Please try again.");
                }
            })
            .catch((error) => {
                // Handle network or other errors
                console.error("Error clearing the file:", error);
                alert("An error occurred while clearing the file. Please check your connection and try again.");
            });
    }
});
