// Publications category filtering
document.addEventListener('DOMContentLoaded', function() {
    const categoryFilter = document.getElementById('category-filter');
    const publicationsContainer = document.getElementById('publications-container');
    const publicationEntries = publicationsContainer.querySelectorAll('.publication-entry');

    categoryFilter.addEventListener('change', function() {
        const selectedCategory = this.value;
        
        publicationEntries.forEach(function(entry) {
            const entryCategory = entry.getAttribute('data-category');
            
            if (selectedCategory === 'all' || entryCategory === selectedCategory) {
                entry.style.display = 'block';
            } else {
                entry.style.display = 'none';
            }
        });
    });
});