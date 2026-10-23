// Which copy of the dashboard this is. The local server (webapp/backend) serves this file
// as is; the GitHub Pages build (scripts/build_pages.py) rewrites it with DQ_STATIC = true,
// and the same pages then read a snapshot of the API from data/api/ instead of a server.
window.DQ_STATIC = false;
window.DQ_REPO = "https://github.com/MichelleGun/CMS";
