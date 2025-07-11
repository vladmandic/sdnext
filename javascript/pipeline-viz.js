/**
 * Pipeline Visualization Component
 * Shows the image generation pipeline as a directed graph
 */

class PipelineVisualization {
    constructor(canvasId, controlsId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.controlsId = controlsId;
        
        this.data = null;
        this.selectedNode = null;
        this.hoveredNode = null;
        this.animationId = null;
        this.isRunning = false;
        
        // Layout properties
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this.nodeWidth = 120;
        this.nodeHeight = 60;
        this.levelHeight = 100;
        this.siblingSpacing = 140;
        this.hierarchyIndent = 30;
        
        // Visual properties
        this.colors = {
            pending: '#6b7280',
            running: '#3b82f6',
            completed: '#10b981',
            failed: '#ef4444',
            skipped: '#f59e0b',
            dependency: '#9ca3af',
            hierarchy: '#4f46e5'
        };
        
        this.setupCanvas();
        this.setupEventListeners();
        this.setupControls();
        this.startPolling();
    }
    
    setupCanvas() {
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
        
        window.addEventListener('resize', () => {
            this.canvas.width = container.clientWidth;
            this.canvas.height = container.clientHeight;
            this.render();
        });
    }
    
    setupEventListeners() {
        let isDragging = false;
        let dragStart = { x: 0, y: 0 };
        let lastPan = { x: this.offsetX, y: this.offsetY };
        
        this.canvas.addEventListener('mousedown', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const clickedNode = this.getNodeAtPosition(x, y);
            if (clickedNode) {
                this.selectedNode = clickedNode;
                this.showNodeDetails(clickedNode);
            } else {
                isDragging = true;
                dragStart = { x: x - this.offsetX, y: y - this.offsetY };
                lastPan = { x: this.offsetX, y: this.offsetY };
            }
            this.render();
        });
        
        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            if (isDragging) {
                this.offsetX = x - dragStart.x;
                this.offsetY = y - dragStart.y;
                this.render();
            } else {
                const hoveredNode = this.getNodeAtPosition(x, y);
                if (hoveredNode !== this.hoveredNode) {
                    this.hoveredNode = hoveredNode;
                    this.render();
                }
            }
        });
        
        this.canvas.addEventListener('mouseup', () => {
            isDragging = false;
        });
        
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const scaleFactor = e.deltaY > 0 ? 0.9 : 1.1;
            const newScale = Math.max(0.1, Math.min(3, this.scale * scaleFactor));
            
            if (newScale !== this.scale) {
                this.offsetX = x - (x - this.offsetX) * (newScale / this.scale);
                this.offsetY = y - (y - this.offsetY) * (newScale / this.scale);
                this.scale = newScale;
                this.render();
            }
        });
    }
    
    setupControls() {
        const controls = document.getElementById(this.controlsId);
        if (!controls) return;
        
        controls.innerHTML = `
            <div class="pipeline-controls">
                <button id="resetView" class="btn btn-secondary">Reset View</button>
                <button id="autoLayout" class="btn btn-secondary">Auto Layout</button>
                <button id="toggleHierarchy" class="btn btn-secondary">Toggle Hierarchy</button>
                <label class="checkbox-label">
                    <input type="checkbox" id="showCompleted" checked> Show Completed
                </label>
                <label class="checkbox-label">
                    <input type="checkbox" id="showSubOps" checked> Show Sub-Operations
                </label>
                <div class="status-info">
                    <span id="pipelineStatus">Status: Unknown</span>
                    <span id="totalDuration">Duration: --</span>
                </div>
            </div>
        `;
        
        document.getElementById('resetView').addEventListener('click', () => {
            this.resetView();
        });
        
        document.getElementById('autoLayout').addEventListener('click', () => {
            this.autoLayout();
        });
        
        document.getElementById('toggleHierarchy').addEventListener('click', () => {
            this.hierarchyMode = !this.hierarchyMode;
            this.render();
        });
        
        document.getElementById('showCompleted').addEventListener('change', (e) => {
            this.showCompleted = e.target.checked;
            this.render();
        });
        
        document.getElementById('showSubOps').addEventListener('change', (e) => {
            this.showSubOps = e.target.checked;
            this.render();
        });
    }
    
    calculateHierarchicalLayout() {
        if (!this.data || !this.data.nodes) return {};
        
        const layout = {};
        const levels = {};
        const nodesByParent = {};
        
        // Group nodes by parent
        this.data.nodes.forEach(node => {
            const parent = node.parent || 'root';
            if (!nodesByParent[parent]) {
                nodesByParent[parent] = [];
            }
            nodesByParent[parent].push(node);
        });
        
        // Calculate positions recursively
        const calculateNodePosition = (nodeId, level = 0, parentX = 0, siblingIndex = 0) => {
            const node = this.data.nodes.find(n => n.id === nodeId);
            if (!node) return { x: parentX, y: level * this.levelHeight };
            
            // Skip if already positioned
            if (layout[nodeId]) return layout[nodeId];
            
            const children = nodesByParent[nodeId] || [];
            const isRoot = level === 0;
            
            // Calculate horizontal position
            let x;
            if (isRoot) {
                x = siblingIndex * (this.nodeWidth + this.siblingSpacing);
            } else {
                x = parentX + (siblingIndex - children.length / 2) * (this.nodeWidth + this.siblingSpacing);
            }
            
            const y = level * this.levelHeight;
            
            layout[nodeId] = { x, y, level, children: children.length };
            
            // Position children
            children.forEach((child, index) => {
                calculateNodePosition(child.id, level + 1, x, index);
            });
            
            return layout[nodeId];
        };
        
        // Start with root nodes
        const rootNodes = this.data.nodes.filter(n => !n.parent);
        rootNodes.forEach((node, index) => {
            calculateNodePosition(node.id, 0, 0, index);
        });
        
        return layout;
    }
    
    render() {
        if (!this.data) return;
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.save();
        
        // Apply transformations
        this.ctx.translate(this.offsetX, this.offsetY);
        this.ctx.scale(this.scale, this.scale);
        
        const layout = this.calculateHierarchicalLayout();
        
        // Draw edges first
        this.drawEdges(layout);
        
        // Draw nodes
        this.drawNodes(layout);
        
        this.ctx.restore();
        
        // Draw UI elements
        this.drawLegend();
        this.updateStatusInfo();
    }
    
    drawEdges(layout) {
        if (!this.data.edges) return;
        
        this.data.edges.forEach(edge => {
            const fromPos = layout[edge.from];
            const toPos = layout[edge.to];
            
            if (!fromPos || !toPos) return;
            
            const fromX = fromPos.x + this.nodeWidth / 2;
            const fromY = fromPos.y + this.nodeHeight;
            const toX = toPos.x + this.nodeWidth / 2;
            const toY = toPos.y;
            
            this.ctx.beginPath();
            this.ctx.strokeStyle = this.colors[edge.type] || this.colors.dependency;
            this.ctx.lineWidth = edge.type === 'hierarchy' ? 2 : 1;
            
            if (edge.type === 'hierarchy') {
                // Draw hierarchical connection with parent-child styling
                this.ctx.setLineDash([5, 5]);
                this.ctx.moveTo(fromX, fromY);
                this.ctx.lineTo(toX, toY);
            } else {
                // Draw dependency connection
                this.ctx.setLineDash([]);
                this.ctx.moveTo(fromX, fromY);
                this.ctx.bezierCurveTo(
                    fromX, fromY + 20,
                    toX, toY - 20,
                    toX, toY
                );
            }
            
            this.ctx.stroke();
            this.ctx.setLineDash([]);
            
            // Draw arrowhead
            const angle = Math.atan2(toY - fromY, toX - fromX);
            const arrowSize = 8;
            this.ctx.beginPath();
            this.ctx.moveTo(toX, toY);
            this.ctx.lineTo(
                toX - arrowSize * Math.cos(angle - Math.PI / 6),
                toY - arrowSize * Math.sin(angle - Math.PI / 6)
            );
            this.ctx.lineTo(
                toX - arrowSize * Math.cos(angle + Math.PI / 6),
                toY - arrowSize * Math.sin(angle + Math.PI / 6)
            );
            this.ctx.closePath();
            this.ctx.fill();
        });
    }
    
    drawNodes(layout) {
        if (!this.data.nodes) return;
        
        this.data.nodes.forEach(node => {
            const pos = layout[node.id];
            if (!pos) return;
            
            // Skip if filtering
            if (!this.showCompleted && node.status === 'completed') return;
            if (!this.showSubOps && node.parent) return;
            
            const x = pos.x;
            const y = pos.y;
            const isSelected = this.selectedNode && this.selectedNode.id === node.id;
            const isHovered = this.hoveredNode && this.hoveredNode.id === node.id;
            
            // Draw node background
            this.ctx.fillStyle = this.colors[node.status] || this.colors.pending;
            this.ctx.strokeStyle = isSelected ? '#ffffff' : (isHovered ? '#d1d5db' : '#374151');
            this.ctx.lineWidth = isSelected ? 3 : (isHovered ? 2 : 1);
            
            // Node shape varies by type
            if (node.parent) {
                // Sub-operations are smaller and rounded
                const width = this.nodeWidth * 0.8;
                const height = this.nodeHeight * 0.6;
                const cornerRadius = 8;
                
                this.ctx.beginPath();
                this.ctx.roundRect(x + (this.nodeWidth - width) / 2, y + (this.nodeHeight - height) / 2, width, height, cornerRadius);
                this.ctx.fill();
                this.ctx.stroke();
            } else {
                // Main operations are larger rectangles
                this.ctx.fillRect(x, y, this.nodeWidth, this.nodeHeight);
                this.ctx.strokeRect(x, y, this.nodeWidth, this.nodeHeight);
            }
            
            // Draw node text
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = node.parent ? '12px sans-serif' : '14px sans-serif';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            
            const textX = x + this.nodeWidth / 2;
            const textY = y + this.nodeHeight / 2;
            
            // Truncate long names
            let displayName = node.name;
            if (displayName.length > 12) {
                displayName = displayName.substring(0, 10) + '...';
            }
            
            this.ctx.fillText(displayName, textX, textY - 5);
            
            // Draw timing info
            if (node.duration) {
                this.ctx.font = '10px sans-serif';
                this.ctx.fillText(`${node.duration.toFixed(2)}s`, textX, textY + 8);
            }
            
            // Draw status indicator
            if (node.status === 'running') {
                this.drawRunningIndicator(x + this.nodeWidth - 12, y + 8);
            } else if (node.status === 'failed') {
                this.drawFailureIndicator(x + this.nodeWidth - 12, y + 8);
            }
        });
    }
    
    drawRunningIndicator(x, y) {
        const time = Date.now() / 1000;
        const opacity = 0.5 + 0.5 * Math.sin(time * 4);
        
        this.ctx.save();
        this.ctx.globalAlpha = opacity;
        this.ctx.fillStyle = '#ffffff';
        this.ctx.beginPath();
        this.ctx.arc(x, y, 4, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.restore();
    }
    
    drawFailureIndicator(x, y) {
        this.ctx.save();
        this.ctx.strokeStyle = '#ffffff';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(x - 3, y - 3);
        this.ctx.lineTo(x + 3, y + 3);
        this.ctx.moveTo(x - 3, y + 3);
        this.ctx.lineTo(x + 3, y - 3);
        this.ctx.stroke();
        this.ctx.restore();
    }
    
    drawLegend() {
        const legendItems = [
            { color: this.colors.pending, text: 'Pending' },
            { color: this.colors.running, text: 'Running' },
            { color: this.colors.completed, text: 'Completed' },
            { color: this.colors.failed, text: 'Failed' },
            { color: this.colors.skipped, text: 'Skipped' }
        ];
        
        const legendX = 10;
        const legendY = 10;
        const itemHeight = 20;
        
        this.ctx.save();
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        this.ctx.fillRect(legendX - 5, legendY - 5, 120, legendItems.length * itemHeight + 10);
        
        legendItems.forEach((item, index) => {
            const y = legendY + index * itemHeight;
            
            this.ctx.fillStyle = item.color;
            this.ctx.fillRect(legendX, y, 15, 15);
            
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = '12px sans-serif';
            this.ctx.textAlign = 'left';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText(item.text, legendX + 20, y + 7);
        });
        
        this.ctx.restore();
    }
    
    updateStatusInfo() {
        if (!this.data) return;
        
        const statusElement = document.getElementById('pipelineStatus');
        const durationElement = document.getElementById('totalDuration');
        
        if (statusElement) {
            statusElement.textContent = `Status: ${this.data.status || 'Unknown'}`;
        }
        
        if (durationElement && this.data.total_duration) {
            durationElement.textContent = `Duration: ${this.data.total_duration.toFixed(2)}s`;
        }
    }
    
    getNodeAtPosition(x, y) {
        if (!this.data || !this.data.nodes) return null;
        
        const layout = this.calculateHierarchicalLayout();
        const transformedX = (x - this.offsetX) / this.scale;
        const transformedY = (y - this.offsetY) / this.scale;
        
        for (const node of this.data.nodes) {
            const pos = layout[node.id];
            if (!pos) continue;
            
            const nodeX = pos.x;
            const nodeY = pos.y;
            
            if (transformedX >= nodeX && transformedX <= nodeX + this.nodeWidth &&
                transformedY >= nodeY && transformedY <= nodeY + this.nodeHeight) {
                return node;
            }
        }
        
        return null;
    }
    
    showNodeDetails(node) {
        const details = document.getElementById('nodeDetails');
        if (!details) return;
        
        let detailsHTML = `
            <div class="node-details">
                <h3>${node.name}</h3>
                <p><strong>Status:</strong> ${node.status}</p>
                <p><strong>ID:</strong> ${node.id}</p>
        `;
        
        if (node.parent) {
            detailsHTML += `<p><strong>Parent:</strong> ${node.parent}</p>`;
        }
        
        if (node.children && node.children.length > 0) {
            detailsHTML += `<p><strong>Children:</strong> ${node.children.length}</p>`;
        }
        
        if (node.duration) {
            detailsHTML += `<p><strong>Duration:</strong> ${node.duration.toFixed(2)}s</p>`;
        }
        
        if (node.dependencies && node.dependencies.length > 0) {
            detailsHTML += `<p><strong>Dependencies:</strong> ${node.dependencies.join(', ')}</p>`;
        }
        
        if (node.details) {
            detailsHTML += '<h4>Details:</h4><ul>';
            for (const [key, value] of Object.entries(node.details)) {
                detailsHTML += `<li><strong>${key}:</strong> ${value}</li>`;
            }
            detailsHTML += '</ul>';
        }
        
        detailsHTML += '</div>';
        details.innerHTML = detailsHTML;
    }
    
    resetView() {
        this.scale = 1;
        this.offsetX = 50;
        this.offsetY = 50;
        this.selectedNode = null;
        this.hoveredNode = null;
        this.render();
    }
    
    autoLayout() {
        if (!this.data || !this.data.nodes) return;
        
        const layout = this.calculateHierarchicalLayout();
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        
        Object.values(layout).forEach(pos => {
            minX = Math.min(minX, pos.x);
            maxX = Math.max(maxX, pos.x + this.nodeWidth);
            minY = Math.min(minY, pos.y);
            maxY = Math.max(maxY, pos.y + this.nodeHeight);
        });
        
        const contentWidth = maxX - minX;
        const contentHeight = maxY - minY;
        const scaleX = (this.canvas.width - 100) / contentWidth;
        const scaleY = (this.canvas.height - 100) / contentHeight;
        
        this.scale = Math.min(scaleX, scaleY, 1);
        this.offsetX = (this.canvas.width - contentWidth * this.scale) / 2 - minX * this.scale;
        this.offsetY = (this.canvas.height - contentHeight * this.scale) / 2 - minY * this.scale;
        
        this.render();
    }
    
    async fetchData() {
        try {
            const response = await fetch('/sdapi/v1/pipeline-viz');
            if (response.ok) {
                this.data = await response.json();
                this.render();
                
                // Auto-start animation if pipeline is running
                if (this.data.status === 'running' && !this.isRunning) {
                    this.startAnimation();
                } else if (this.data.status !== 'running' && this.isRunning) {
                    this.stopAnimation();
                }
            }
        } catch (error) {
            console.error('Failed to fetch pipeline data:', error);
        }
    }
    
    startAnimation() {
        if (this.isRunning) return;
        
        this.isRunning = true;
        const animate = () => {
            if (this.isRunning) {
                this.render();
                this.animationId = requestAnimationFrame(animate);
            }
        };
        animate();
    }
    
    stopAnimation() {
        this.isRunning = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
    
    startPolling() {
        this.fetchData();
        setInterval(() => this.fetchData(), 1000);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const viz = new PipelineVisualization('pipelineCanvas', 'pipelineControls');
    window.pipelineViz = viz; // Make it globally accessible
}); 