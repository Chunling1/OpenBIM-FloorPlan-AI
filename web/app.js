/**
 * BIM FloorPlan AI - Frontend App
 */

(function () {
    'use strict';

    // ======== DOM refs ========
    const $ = (sel) => document.querySelector(sel);
    const uploadZone = $('#uploadZone');
    const uploadInner = $('#uploadInner');
    const previewThumb = $('#previewThumb');
    const previewImg = $('#previewImg');
    const btnClear = $('#btnClear');
    const fileInput = $('#fileInput');
    const btnPredict = $('#btnPredict');
    const emptyState = $('#emptyState');
    const loadingOverlay = $('#loadingOverlay');
    const loadingTimer = $('#loadingTimer');
    const resultView = $('#resultView');
    const statsPanel = $('#statsPanel');
    const statBars = $('#statBars');
    const statMeta = $('#statMeta');
    const geoPanel = $('#geoPanel');
    const geoGrid = $('#geoGrid');
    const deviceBadge = $('#deviceBadge');
    const splitHandle = $('#splitHandle');
    const viewerContainer = $('#viewerContainer');

    let selectedFile = null;
    let timerInterval = null;

    // ======== Init ========
    init();

    function init() {
        setupParticles();
        setupUpload();
        setupRadio();
        setupTabs();
        setupSplit();
        checkHealth();
    }

    // ======== Background Particles ========
    function setupParticles() {
        const canvas = $('#particles-bg');
        const ctx = canvas.getContext('2d');
        let w, h, particles = [];

        function resize() {
            w = canvas.width = window.innerWidth;
            h = canvas.height = window.innerHeight;
        }
        resize();
        window.addEventListener('resize', resize);

        for (let i = 0; i < 40; i++) {
            particles.push({
                x: Math.random() * w,
                y: Math.random() * h,
                r: Math.random() * 1.5 + 0.5,
                dx: (Math.random() - 0.5) * 0.3,
                dy: (Math.random() - 0.5) * 0.3,
                opacity: Math.random() * 0.3 + 0.05,
            });
        }

        function draw() {
            ctx.clearRect(0, 0, w, h);
            for (const p of particles) {
                p.x += p.dx; p.y += p.dy;
                if (p.x < 0) p.x = w;
                if (p.x > w) p.x = 0;
                if (p.y < 0) p.y = h;
                if (p.y > h) p.y = 0;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(99, 102, 241, ${p.opacity})`;
                ctx.fill();
            }
            // draw faint connections
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 150) {
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.strokeStyle = `rgba(99, 102, 241, ${0.03 * (1 - dist / 150)})`;
                        ctx.lineWidth = 0.5;
                        ctx.stroke();
                    }
                }
            }
            requestAnimationFrame(draw);
        }
        draw();
    }

    // ======== Upload ========
    function setupUpload() {
        // Click to open file dialog
        uploadInner.addEventListener('click', () => fileInput.click());

        // Drag & drop
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });
        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) handleFile(files[0]);
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
        });

        btnClear.addEventListener('click', (e) => {
            e.stopPropagation();
            clearFile();
        });

        btnPredict.addEventListener('click', runPrediction);
    }

    function handleFile(file) {
        if (!file.type.startsWith('image/')) return;
        selectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            uploadInner.style.display = 'none';
            previewThumb.style.display = 'block';
            btnPredict.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function clearFile() {
        selectedFile = null;
        fileInput.value = '';
        uploadInner.style.display = 'flex';
        previewThumb.style.display = 'none';
        btnPredict.disabled = true;
    }

    // ======== Radio ========
    function setupRadio() {
        document.querySelectorAll('.radio-item').forEach(item => {
            item.addEventListener('click', () => {
                document.querySelectorAll('.radio-item').forEach(r => r.classList.remove('active'));
                item.classList.add('active');
                item.querySelector('input').checked = true;
            });
        });
    }

    // ======== Tabs ========
    function setupTabs() {
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                const view = tab.dataset.view;
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                document.querySelectorAll('.view-pane').forEach(p => p.classList.remove('active'));
                document.querySelector(`.view-pane[data-view="${view}"]`).classList.add('active');
            });
        });
    }

    // ======== Split Compare ========
    function setupSplit() {
        let isDragging = false;

        splitHandle.addEventListener('mousedown', (e) => {
            isDragging = true;
            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const splitPane = $('#splitCompare');
            const rect = splitPane.getBoundingClientRect();
            let x = (e.clientX - rect.left) / rect.width;
            x = Math.max(0.1, Math.min(0.9, x));
            const pct = x * 100;
            splitHandle.style.left = pct + '%';
            splitPane.querySelector('.split-right').style.left = pct + '%';
        });

        document.addEventListener('mouseup', () => { isDragging = false; });
    }

    // ======== Health Check ========
    async function checkHealth() {
        try {
            const res = await fetch('/api/health');
            const data = await res.json();
            deviceBadge.textContent = data.device || '—';
        } catch {
            deviceBadge.textContent = 'offline';
        }
    }

    // ======== Run Prediction ========
    async function runPrediction() {
        if (!selectedFile) return;

        // Show loading
        emptyState.style.display = 'none';
        resultView.style.display = 'none';
        loadingOverlay.style.display = 'flex';
        btnPredict.disabled = true;
        btnPredict.classList.add('loading');
        btnPredict.innerHTML = '<span class="shimmer"></span><span>Extracting...</span>';

        // Timer
        let elapsed = 0;
        loadingTimer.textContent = '0.0s';
        timerInterval = setInterval(() => {
            elapsed += 0.1;
            loadingTimer.textContent = elapsed.toFixed(1) + 's';
        }, 100);

        // Build form
        const preprocessing = document.querySelector('input[name="preprocess"]:checked').value;
        const form = new FormData();
        form.append('file', selectedFile);
        form.append('preprocessing', preprocessing);

        try {
            const res = await fetch('/api/predict', { method: 'POST', body: form });
            const data = await res.json();

            clearInterval(timerInterval);
            loadingOverlay.style.display = 'none';

            if (data.success) {
                showResults(data);
            } else {
                alert('Inference Failed: ' + (data.error || 'Unknown error'));
            }
        } catch (err) {
            clearInterval(timerInterval);
            loadingOverlay.style.display = 'none';
            alert('Request Failed: ' + err.message);
        }

        btnPredict.disabled = false;
        btnPredict.classList.remove('loading');
        btnPredict.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M9 2a7 7 0 105.3 11.7l3.7 3.7" stroke-linecap="round"/>
            </svg>
            <span>Extract Elements 提取构件</span>`;
    }

    // ======== Show Results ========
    function showResults(data) {
        resultView.style.display = 'flex';
        resultView.classList.add('fade-in');

        // Images
        const overlayUrl = 'data:image/jpeg;base64,' + data.images.overlay;
        const maskUrl = 'data:image/png;base64,' + data.images.mask;
        const origUrl = 'data:image/jpeg;base64,' + data.images.original;

        $('#imgOverlay').src = overlayUrl;
        $('#imgMask').src = maskUrl;
        $('#imgOrigSplit').src = origUrl;
        $('#imgOverlaySplit').src = overlayUrl;

        // Stats Panel
        statsPanel.style.display = 'block';
        statsPanel.classList.add('fade-in');

        const items = [
            { key: 'wall', name: '墙体 Wall', color: '#e74c3c' },
            { key: 'window', name: '窗户 Win', color: '#3498db' },
            { key: 'door', name: '门 Door', color: '#2ecc71' },
        ];

        statBars.innerHTML = items.map(item => {
            const s = data.stats[item.key];
            // 用非背景像素的最大值作为100%来显示bar
            const maxPct = Math.max(...items.map(i => data.stats[i.key].percentage), 1);
            const barWidth = (s.percentage / maxPct * 100).toFixed(1);
            return `
                <div class="stat-row">
                    <span class="stat-dot" style="background:${item.color}"></span>
                    <span class="stat-name">${item.name}</span>
                    <div class="stat-bar-track">
                        <div class="stat-bar-fill" style="background:${item.color};width:${barWidth}%"></div>
                    </div>
                    <span class="stat-pct">${s.percentage}%</span>
                </div>`;
        }).join('');

        // Animate bars
        setTimeout(() => {
            statBars.querySelectorAll('.stat-bar-fill').forEach(bar => {
                const w = bar.style.width;
                bar.style.width = '0';
                requestAnimationFrame(() => { bar.style.width = w; });
            });
        }, 50);

        statMeta.innerHTML = `
            <span class="stat-meta-item">Size: <strong>${data.image_size[0]}×${data.image_size[1]}</strong></span>
            <span class="stat-meta-item">Time: <strong>${data.elapsed_sec}s</strong></span>
            <span class="stat-meta-item">Model: <strong>${data.model_info.name}</strong></span>
        `;

        // Geo Panel
        geoPanel.style.display = 'block';
        geoPanel.classList.add('fade-in');

        const gs = data.geometry_summary;
        geoGrid.innerHTML = `
            <div class="geo-card">
                <div class="geo-count" style="color:#e74c3c">${gs.walls}</div>
                <div class="geo-label">Walls</div>
            </div>
            <div class="geo-card">
                <div class="geo-count" style="color:#3498db">${gs.windows}</div>
                <div class="geo-label">Windows</div>
            </div>
            <div class="geo-card">
                <div class="geo-count" style="color:#2ecc71">${gs.doors}</div>
                <div class="geo-label">Doors</div>
            </div>
        `;

        // Switch to overlay tab
        document.querySelector('.tab[data-view="overlay"]').click();
    }

})();
