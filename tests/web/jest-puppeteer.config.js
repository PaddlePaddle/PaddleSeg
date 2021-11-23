// jest-puppeteer.config.js
module.exports = {
    launch: {
        headless: false,
        product: 'chrome'
    },
    browserContext: 'default',
    server: {
        command: 'python3 -m http.server 9898',
        port: 9898,
        launchTimeout: 10000,
        debug: true
    }
};