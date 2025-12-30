/**
 * 设置一个通用的按钮切换表格/内容的功能。
 *
 * @param {string} btn1Id - 负责显示 table1 的按钮的 DOM ID (例如 'rag-multi-btn')
 * @param {string} btn2Id - 负责显示 table2 的按钮的 DOM ID (例如 'rag-single-btn')
 * @param {string} table1Id - 要显示的第一个表格/内容的 DOM ID (例如 'rag-multi-table')
 * @param {string} table2Id - 要显示的第二个表格/内容的 DOM ID (例如 'rag-single-table')
 */
function setupTabToggle(btn1Id, btn2Id, table1Id, table2Id) {
    // 确保 DOM 元素存在
    const btn1 = document.getElementById(btn1Id);
    const btn2 = document.getElementById(btn2Id);
    const table1 = document.getElementById(table1Id);
    const table2 = document.getElementById(table2Id);

    if (!btn1 || !btn2 || !table1 || !table2) {
        console.error('setupTabToggle 错误：未找到所有必需的 DOM 元素。', { btn1Id, btn2Id, table1Id, table2Id });
        return;
    }

    /**
     * 内部切换逻辑
     * @param {string} showId - 要显示的表格的 ID
     */
    function toggleContent(showId) {
        if (showId === table1Id) {
            // 切换到 table1
            btn1.classList.add('active');
            btn2.classList.remove('active');
            table1.style.display = 'table';
            table2.style.display = 'none';
        } else if (showId === table2Id) {
            // 切换到 table2
            btn1.classList.remove('active');
            btn2.classList.add('active');
            table1.style.display = 'none';
            table2.style.display = 'table';
        }
    }

    // 绑定事件监听器
    btn1.addEventListener('click', () => toggleContent(table1Id));
    btn2.addEventListener('click', () => toggleContent(table2Id));

    // 确保初始状态（如果 btn1 有 active 类，则显示 table1）
    if (btn1.classList.contains('active') && table2.style.display !== 'none') {
        // 如果 btn1 初始被选中，但 table2 却显示着，则强制切换到 table1
        toggleContent(table1Id);
    } else if (btn2.classList.contains('active') && table1.style.display !== 'none') {
        // 如果 btn2 初始被选中，但 table1 却显示着，则强制切换到 table2
        toggleContent(table2Id);
    }
}


// =====================================================================
// 应用该函数：在 DOM 加载完成后，调用该函数来设置 RAG 排行榜的切换功能
// =====================================================================

document.addEventListener('DOMContentLoaded', function() {
   
    // 使用新的通用函数设置切换功能
    setupTabToggle('rag-multi-btn','rag-single-btn', 'rag-multi-table', 'rag-single-table');
    setupTabToggle('unique-ungrouped-btn', 'unique-grouped-btn', 'unique-ungrouped-table', 'unique-grouped-table');
    setupTabToggle('ex-ungrouped-btn', 'ex-grouped-btn', 'ex-ungrouped-table', 'ex-grouped-table');

    /* 注意：在您的原始 HTML 中，按钮使用了内联的 onclick 事件：
    <button type="button" class="btn btn-f" id="rag-multi-btn" onclick="showRAGTable('multi')">
    
    使用此新函数后，您可以移除 HTML 中的 onclick 属性，因为事件已经通过 addEventListener 绑定。
    如果保留 onclick，可能会导致两次执行切换逻辑。
    */
});