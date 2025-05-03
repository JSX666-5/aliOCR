document.querySelectorAll('.upload-input').forEach(input => {
    input.addEventListener('change', function(e) {
        const type = this.dataset.type;
        const box = this.closest('.upload-box');
        const preview = box.querySelector('.preview');
        const results = box.querySelector('.results');

        const file = e.target.files[0];
        if (!file) return;

        // 显示预览图
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.innerHTML = `<img src="${e.target.result}" alt="预览">`;
        }
        reader.readAsDataURL(file);

        // 添加加载状态
        box.classList.add('loading');

        // 上传文件
        const formData = new FormData();
        formData.append('file', file);

        let endpoint = '';
        switch(type) {
            case 'no': endpoint = '/upload/no'; break;
            case 'select': endpoint = '/upload/select'; break;
            case 'program': endpoint = '/upload/program'; break;
        }

        fetch(endpoint, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                results.innerHTML = `<div class="error">错误：${data.error}</div>`;
            } else {
                results.innerHTML = data.results
                    .map(text => `<div class="result-item">${text}</div>`)
                    .join('');
            }
        })
        .catch(error => {
            results.innerHTML = `<div class="error">请求失败：${error.message}</div>`;
        })
        .finally(() => {
            box.classList.remove('loading');
        });
    });
});