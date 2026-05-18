// Tab Navigation for LSTM Simulator
document.addEventListener('DOMContentLoaded', () => {
    const tabBtns = document.querySelectorAll('.tab-btn[data-tab]');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.tab;

            // Remove active class from all buttons and contents
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Add active class to clicked button and corresponding content
            btn.classList.add('active');
            const el = document.getElementById(target);
            if (el) el.classList.add('active');

            // Re-trigger canvas sizing after tab becomes visible
            requestAnimationFrame(() => {
                window.dispatchEvent(new Event('resize'));
            });
        });
    });
});
