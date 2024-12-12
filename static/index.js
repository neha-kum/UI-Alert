const sideMenu = document.querySelector('aside');
const menuBtn = document.getElementById('menu-btn');
const closeBtn = document.getElementById('close-btn');

const darkMode = document.querySelector('.dark-mode');

menuBtn.addEventListener('click', () => {
    sideMenu.style.display = 'block';
});

closeBtn.addEventListener('click', () => {
    sideMenu.style.display = 'none';
});

darkMode.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode-variables');
    darkMode.querySelector('span:nth-child(1)').classList.toggle('active');
    darkMode.querySelector('span:nth-child(2)').classList.toggle('active');
})

// Utkarsh Changes
document.addEventListener("DOMContentLoaded", function () {
    // Get the container where the existing projects will be displayed
    const existingProjects = document.getElementById("existingProjects");

    // Fetch the list of existing projects from the Flask server
    fetch('/get_existing_projects')
        .then(response => response.json())
        .then(data => {
            if (data.projects) {
                data.projects.forEach((project, index) => {
                    const projectDiv = document.createElement('div');
                    projectDiv.classList.add('projects');
                    // Dynamically construct the URL
                    const projectUrl = `/projects/${encodeURIComponent(project)}`;
                    projectDiv.innerHTML = `
                    <a href="${projectUrl}">
                        <h1>${project}</h1>
                    </a>
                    <h3>Found ${data.files[index]} Images & Masks</h3>
                `;
                    existingProjects.appendChild(projectDiv);
                });
            } else {
                console.error('No projects found.');
            }
        })
        .catch(error => {
            console.error('Error fetching projects:', error);
        });

});
