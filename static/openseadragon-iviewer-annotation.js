
class LayerFIFOQueue {
    constructor(layer, capacity = 64, KeysBackingArrayType = Uint32Array) {
        this.capacity = capacity;
        this.layer = layer;
        this.childIndices = new Set();
        this.length = 0;
    }

    add(child) {
        if (this.childIndices.has(child.ann.id)) {
            return;
        }

        if (this.isFull()) {
            let pop_child = this.layer.children[0];  // O(n) not O(1)
            this.remove(pop_child);
        }
//         let area = (child.ann['x1'] - child.ann['x0']) * (child.ann['y1'] - child.ann['y0']);
        this.layer.add(child);
        this.childIndices.add(child.ann.id);
        this.length++;
    }

    remove(child) {
        this.childIndices.delete(child.ann.id);
        child.destroy();
        this.length--;
    }
    
    children() {
        return this.layer.children;
    }

    destroyChildren() {
        this.layer.destroyChildren();
        this.length = 0;
        this.childIndices.clear();
    }

    isEmpty() {
        return this.length === 0;
    }

    isFull() {
        return this.length === this.capacity;
    }
}

class ColorPalette {
    static instanceCount = 0;
    static paletteInstances = {};

    constructor(container, colorPalette = {}, defaultColor = '#E8E613') {
        this.annotationLayers = [];
        this.colorPalette = colorPalette;
        this.defaultColor = defaultColor;
        this.isDragging = false;
        this.offsetX = 0;
        this.offsetY = 0;
        this.instanceId = `colorPopUp_${++ColorPalette.instanceCount}`;
        ColorPalette.paletteInstances[this.instanceId] = this;
        this.tempColorPalette = { ...colorPalette };  // Temporary storage for changes
        this.hideLabels = new Set();
        this.legend = document.getElementById("color-legend");

        // register annotationLayer and container
        // use arrow, don't use function (this in function refer to container not instance)
        container.onclick = () => {
            this.openColorPopUp(container);  // 'this' refers to the ColorPalette instance
        };
        this.container = container;

        this.createColorPopUp();
        this.setupDragListeners();

        for (const [label, color] of Object.entries(this.tempColorPalette)) {
            this.createLegend(label, color); //also generate legend at same time
        }
    }

    addLayers(annotationLayers) {
        this.annotationLayers.push(...annotationLayers);
    }

    getColor(label, defaultColor=null) {
        let color = this.colorPalette[label] || defaultColor || this.defaultColor;
        
        if (this.hideLabels.has(label)) {
            return {'border': color + '00', 'face': color + '00', 
                'box-shadow': `0 0 0 10px ${color}, inset 0 0 0 10px ${color}` }
        } else {
            return {'border': color + 'ff', 'face': color + '35', 
                'box-shadow': `0 0 0 10px ${color}, inset 0 0 0 10px ${color}` }
        }
    }

    labels() {
        return Object.keys(this.colorPalette);
    }

    createColorPopUp() {
        const colorPopUp = document.createElement('div');
        colorPopUp.classList.add('colorPopUp');
        colorPopUp.id = this.instanceId;
        colorPopUp.innerHTML = `
            <div>Label Colors</div>
            <div class="colorList"></div>
            <div class="popUpButtons">
                <button onclick="ColorPalette.paletteInstances['${this.instanceId}'].addNewLabel()">Add Label</button>
                <button onclick="ColorPalette.paletteInstances['${this.instanceId}'].deleteSelectedLabels()">Delete Selected Labels</button>
                <button onclick="ColorPalette.paletteInstances['${this.instanceId}'].confirmChanges()">Confirm</button>
                <button onclick="ColorPalette.paletteInstances['${this.instanceId}'].cancelChanges()">Cancel</button>
            </div>
        `;
        document.body.appendChild(colorPopUp);
        this.colorPopUp = colorPopUp;
        this.colorList = colorPopUp.querySelector('.colorList');
        this.colorPopUp.style.display = "none";
    }

    openColorPopUp(button) {
        this.tempColorPalette = { ...this.colorPalette };  // Reset temporary palette
        this.generateColorPopUp();
        const rect = {
            top: button.offsetTop,
            left: button.offsetLeft,
            bottom: button.offsetTop + button.offsetHeight,
            right: button.offsetLeft + button.offsetWidth
        };
        this.colorPopUp.style.left = `${rect.left}px`;
        this.colorPopUp.style.top = `${rect.bottom}px`;
        this.colorPopUp.style.display = "block";
    }

    closeColorPopUp() {
        this.colorPopUp.style.display = "none";
    }

    generateColorPopUp() {
        this.colorList.innerHTML = ''; // Clear previous content
        this.legend.innerHTML = '';
        for (const [label, color] of Object.entries(this.tempColorPalette)) {
            this.createLabelRow(label, color);
            this.createLegend(label, color); //also generate legend at same time
        }
    }

    createLegend(label, color) {
        console.log("create legend")
        const tagColor = document.createElement('div');
        tagColor.classList.add('legend-box');
        tagColor.style.backgroundColor = color;
        const tag = document.createElement('span');
        tag.textContent = label;
        tag.classList.add('legend-text'); 
        this.legend.appendChild(tagColor);
        this.legend.appendChild(tag);
     }

    createLabelRow(label, color) {
        const labelRow = document.createElement('div');
        labelRow.classList.add('label-row');

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.classList.add('label-checkbox');

        const labelInput = document.createElement('input');
        labelInput.type = 'text';
        labelInput.value = label;
        labelInput.classList.add('label-name');

        labelInput.oninput = () => {
            let newLabel = labelInput.value.trim();
            newLabel = newLabel.replace(/[-\s]/g, '_');
            labelInput.value = newLabel;
            this.updateLabelInTempMap(label, newLabel, colorInput.value);
            label = newLabel;
        }

        const colorInput = document.createElement('input');
        colorInput.type = 'color';
        colorInput.value = color;
        colorInput.classList.add('color-input');
        colorInput.oninput = () => {
            this.tempColorPalette[label] = colorInput.value;
        };

        labelRow.appendChild(checkbox);
        labelRow.appendChild(labelInput);
        labelRow.appendChild(colorInput);

        this.colorList.appendChild(labelRow);
    }

    addNewLabel() {
        const newLabel = this.generateUniqueLabel("New_Label");
        const newColor = this.defaultColor;
        this.tempColorPalette[newLabel] = newColor;
        console.log("added new temp color", newLabel)
        this.createLabelRow(newLabel, newColor);
    }

    generateUniqueLabel(baseLabel) {
        let label = baseLabel;
        let counter = 1;
        while (this.tempColorPalette.hasOwnProperty(label)) {
            label = `${baseLabel}_${counter++}`;
        }
        return label;
    }

    deleteSelectedLabels() {
        const checkboxes = this.colorPopUp.querySelectorAll('.label-checkbox');
        checkboxes.forEach(checkbox => {
            if (checkbox.checked) {
                const labelRow = checkbox.parentElement;
                const labelInput = labelRow.querySelector('.label-name');
                const label = labelInput.value;

                delete this.tempColorPalette[label];
                labelRow.remove();
            }
        });
    }

    updateLabelInTempMap(oldLabel, newLabel, color) {
        if (oldLabel !== newLabel && newLabel !== "") {
            delete this.tempColorPalette[oldLabel];
            this.tempColorPalette[newLabel] = color;
        }
    }

    confirmChanges() {
        const labelInputs = document.querySelectorAll('.label-name');
        const labelSet = new Set();
        let hasDuplicate = false;

         // Check for duplicate labels
        labelInputs.forEach((input) => {
            const label = input.value.trim();
            if (labelSet.has(label)) {
                alert(`The label "${label}" is duplicated. Please choose a different label.`);
                hasDuplicate = true;
                return;  // Abort
            }
            labelSet.add(label);  // Add to set if unique
        });

        if (hasDuplicate) {
            return;  
        }

        let deleteKeys = new Set(Object.keys(this.colorPalette).filter(x => !(x in this.tempColorPalette)));
        let addKeys = new Set(Object.keys(this.tempColorPalette).filter(x => !(x in this.colorPalette)));

        // Remove keys in deleteKeys from hideLabels
        for (let key of deleteKeys) {
            this.hideLabels.add(key);
        }

        // Add keys in addKeys to hiddenColors
        for (let key of addKeys) {
            if (this.hideLabels.has(key)) {
                this.hideLabels.delete(key);
            }
        }

        this.colorPalette = { ...this.tempColorPalette };  // Apply changes
        this.closeColorPopUp();
        this.annotationLayers.forEach((layer) => {
            layer.clear();
            layer.draw();
        });        

        // Send the colorPalette data to Laravel
        const imageUUID = imgId; 
        fetch('/save-color-palette', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                // 'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').getAttribute('content')
            },
            body: JSON.stringify({
                image_id: imageUUID,
                color_palette: this.colorPalette,
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log('Palette saved successfully');
            }
        }).catch(error => {
            console.error('Error saving palette:', error);
        })
        .finally(() => {
            // Update the legend regardless of success or failure
            this.legend.innerHTML = '';
            for (const [label, color] of Object.entries(this.tempColorPalette)) {
                this.createLegend(label, color); // generate the legend at the same time
            }
        });

        //update the legend
        this.legend.innerHTML = '';
        for (const [label, color] of Object.entries(this.tempColorPalette)) {
            this.createLegend(label, color); //also generate legend at same time
        }
    }

    cancelChanges() {
        this.closeColorPopUp();  // Discard temporary changes
    }

    setupDragListeners() {
        this.colorPopUp.addEventListener("mousedown", (e) => {
            if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
                this.isDragging = true;
                this.offsetX = e.clientX - parseInt(window.getComputedStyle(this.colorPopUp).left);
                this.offsetY = e.clientY - parseInt(window.getComputedStyle(this.colorPopUp).top);
            }
        });

        document.addEventListener("mousemove", (e) => {
            if (this.isDragging) {
                this.colorPopUp.style.left = `${e.clientX - this.offsetX}px`;
                this.colorPopUp.style.top = `${e.clientY - this.offsetY}px`;
            }
        });

        document.addEventListener("mouseup", () => {
            this.isDragging = false;
        });
    }
}

class IViewerAnnotation {
    static count = 0;
    constructor(viewer, configs) {
//         if (!window.OpenSeadragon) {
//             console.error('[openseadragon-konva-overlay] requires OpenSeadragon');
//             return;
//         }
        this._viewer = viewer;
        this.cfs = configs || {};
        this._id = configs?.id || 'osd-overlaycanvas-' + (++IViewerAnnotation.count);
//         this.cfs.enablePointerEvents = window.PointerEvent != null;
        if (this.cfs?.disableClickToZoom) {
            this._viewer.gestureSettingsMouse.clickToZoom = false;
        }

        // Build canvas div and Konva Stage/Layers
        this._canvasdiv = document.createElement('div');
        this._canvasdiv.setAttribute('id', this._id);
        this._canvasdiv.style.position = 'absolute';
        this._canvasdiv.style.left = 0;
        this._canvasdiv.style.top = 0;
        this._canvasdiv.style.width = '100%';
        this._canvasdiv.style.height = '100%';
        this._viewer.canvas.appendChild(this._canvasdiv);

        this._containerWidth = 0;
        this._containerHeight = 0;
        this.resize();

        // Create a Konva stage in canvas div
        this._konvaStage = new Konva.Stage({
            container: this._id,
//             width: this._viewer.container.clientWidth,
//             height: this._viewer.container.clientHeight,
//             draggable: false,  // Disable draggable because default click is tracked by OSD
//             opacity: 1.0,
        });
        this.registerKonvaActions();

        // Add Konva Layers
        const layerCfgs = this.cfs?.layers || [{'id': 'konvaLayer-0', 'capacity': 512}];
        this.layerQueues = {};
        layerCfgs.forEach((cfg, index) => {
            let id = cfg?.id || `konvaLayer-${index}`;
            let capacity = cfg?.capacity || null;
            this.addLayer(id=id, capacity=capacity);
        });

        // Add a colorPalette to Konva if given.
        let colorPalette = this.cfs?.colorPalette || {'colors': {}};
        if (colorPalette instanceof ColorPalette) {
            this.colorPalette = colorPalette;
            this.colorPalette.addLayers([this]);
        } else {
            let colorPaletteCfgs = colorPalette;
            let colorButton = colorPaletteCfgs?.container || generateButton(viewer);
            this.colorPalette = new ColorPalette(
                colorButton,
                colorPaletteCfgs?.colors || {},
                colorPaletteCfgs?.default || '#E8E613',
            );
            this.colorPalette.addLayers([this]);
        }
        
        // Add Annotorious Layer
        let widgets = this.cfs?.widgets || [];
        this._annotoriousLayer = OpenSeadragon.Annotorious(viewer, {
            locale: 'auto',
            allowEmpty: true,
            widgets: widgets,
        });

        // Add Annotorious SelectorPack and ToolBars
        let toolCfgs = this.cfs?.drawingTools || {'tools': ['point', 'rect', 'polygon', 'circle', 'ellipse', 'freehand']};
        if (toolCfgs) {
            try {
                Annotorious.BetterPolygon(this._annotoriousLayer);
            } catch (error) {  // Package is not available
                console.log("Annotorious.BetterPolygon is not available. Use default polygon.");
            }
            Annotorious.SelectorPack(this._annotoriousLayer, {tools: toolCfgs.tools});
            if (toolCfgs?.container) {
                let barCfgs = {'drawingTools': toolCfgs.tools || toolCfgs?.drawingTools};
                if (toolCfgs?.withMouse) barCfgs['withMouse'] = true;
                if (toolCfgs?.withLabel) barCfgs['withLabel'] = true;
                if (toolCfgs?.withTooltip) barCfgs['withTooltip'] = true;
                if (toolCfgs?.infoElement) barCfgs['infoElement'] = toolCfgs.infoElement;
                console.log(barCfgs);
                Annotorious.Toolbar(this._annotoriousLayer, toolCfgs.container, barCfgs);
            }
        }

        // Add A pool of modifying objects. TODO: use array in future
        this._modifingNode = null;

        this.activeAnnotators = new Set();
        
        // Bind all event functions
        this.render = this.render.bind(this);
        this.renderActive = this.renderActive.bind(this);
    }

    registerKonvaActions() {
        // Update viewport
        this._viewer.addHandler('update-viewport', () => {
            this.resize();
            this.resizeCanvas();
        });

        // Resize the konva overlay when the viewer or window changes size
        this._viewer.addHandler('open', () => {
            this.resize();
            this.resizeCanvas();
        });

        window.addEventListener('resize', () => {
            this.resize();
            this.resizeCanvas();
        });

        // window.addEventListener('scroll', function (event) {
        //     // this.resize();
        //     // this.resizeCanvas();
        //     console.log("scoll down/up")
        //     event.preventDefault();
        // }, { passive: true });
    }

    buildConnections(createDB, getAnnotators, getLabels, insert, read, update, deleteanno, search, stream, count) {
        this.APIs = {
            annoCreateDBAPI: createDB,
            annoGetAnnotators: getAnnotators,
            annoGetLabels: getLabels,
            annoInsertAPI: insert,
            annoReadAPI: read,
            annoUpdateAPI: update,
            annoDeleteAPI: deleteanno,
            annoSearchAPI: search,
            annoStreamAPI: stream,
            annoCountAnnos: count,
        }
        
        //after loading the slide, call api to create its db
        fetch(this.APIs.annoCreateDBAPI, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        }).then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
        }).catch(error => {
            console.error('There was a when creating database:', error);
        });

        this.webSocket = new WebSocket(this.APIs.annoStreamAPI);
//         this._viewer.addHandler('viewport-change', event => {
//             this.render();
//         });
    }

    streamAnnotations(layerQueue, query) {
        // skip message if null, missing(undefined) is allowed
        let tag = true;
        if ('annotator' in query) {
            if (query['annotator'] === null) tag = false;
            if (Array.isArray(query['annotator']) && query['annotator'].length === 0) tag = false;
        }
        if ('label' in query) {
            if (query['label'] === null) tag = false;
            if (Array.isArray(query['label']) && query['label'].length === 0) tag = false;
        }

        if (tag) {
            // this.webSocket.send(JSON.stringify(query));
            sendMessageWithRetry(this.webSocket, query);
            this.webSocket.onmessage = (event) => {
                let ann = JSON.parse(event.data);
                // console.log("return annotation", ann)
                let item = this.createKonvaItem(ann);
                layerQueue.add(item);
            }
//             layerQueue.layer.batchDraw();
        }
    };

    //= streamannotation function
    render(cfgs={}) {
        let bbox = getCurrentViewport(this._viewer);
        let query;
        if (bbox) {
            let windowArea = (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0);
            query = {
                'x0': bbox.x0,
                'y0': bbox.y0,
                'x1': bbox.x1,
                'y1': bbox.y1,
                'min_box_area': windowArea * 0.0001 - 100,
                'max_box_area': windowArea,
                ...cfgs,
            }
        } else {  // if no bbox information in query
            query = {...cfgs};
        }
        // console.log('Search query:', query);
        this.streamAnnotations(this.getLayerQueue(), query);
        // this._konvaStage.draw();
    }

    renderActive() {
        this.render({'annotator': Array.from(this.activeAnnotators)});
    } 

    draw() {
        this._viewer.addHandler('viewport-change', this.renderActive);
    }

    hide() {
        this._viewer.removeHandler('viewport-change', this.renderActive);
    }

    updateAnnotators(annotators) {
        let newAnnotators = new Set(annotators);
//         console.log("Incoming Annotators: ", newAnnotators);

        // Remove annotators
        const deletedIds = new Set([...this.activeAnnotators].filter(x => !newAnnotators.has(x)));
        console.log("deletedids", deletedIds);
        if (deletedIds.size > 0) {
//             console.log("Delete Annotators: ", deletedIds);
            let layerQueue = this.getLayerQueue();
            let popNodes = layerQueue.children().filter(
                child => deletedIds.has(child.ann.annotator)
            );
            popNodes.forEach(node => {
                layerQueue.remove(node);
            });
        }

        // Add annotators
        const addIds = new Set([...newAnnotators].filter(x => !this.activeAnnotators.has(x)));
        console.log("addIds", addIds);
        if (addIds.size > 0) {
//             console.log("Add Annotators: ", addIds);
            this.render({'annotator': Array.from(addIds)});
        }

        this.activeAnnotators = newAnnotators;
//         console.log("Updated Annotators: ", this.activeAnnotators);
    }

    // Add a new annotator into activeAnnotators
    addAnnotator(annotator) {
        if (!this.activeAnnotators.has(annotator)) {
            this.activeAnnotators.add(annotator);
            this.render({'annotator': annotator});
        }
    }

    // Remove a annotator from activeAnnotators
    removeAnnotator(annotator) {
        if (this.activeAnnotators.has(annotator)) {
            this.activeAnnotators.delete(annotator);
            let layerQueue = this.getLayerQueue();
            let popNodes = layerQueue.children().filter(
                child => child.ann.annotator === annotator
            );
            popNodes.forEach(node => {
                layerQueue.remove(node);
            });
        }
    }

    enableEditing(userId) {
        this.userId = userId;

        // Enter modify mode: hide object and create an annotorious object
        this._viewer.addHandler("canvas-click", event => {
            let selected = this._annotoriousLayer.getSelected();
            if (!selected) {  // make sure no existing selection
                // let item = this._konvaStage.getIntersection({ x: event.position.x, y: event.position.y }); // more efficient but can't find small item which is under large item
                let items = this._konvaStage.getAllIntersections({ x: event.position.x, y: event.position.y });
                let smallestValue = Infinity, item=items[0];
                for (let i of items) {
                    // Calculate the absolute differences
                    let xDiff = Math.abs(i.ann.x1 - i.ann.x0);
                    let yDiff = Math.abs(i.ann.y1 - i.ann.y0);

                    // Calculate the product of absolute differences
                    let product = xDiff * yDiff;

                    // Check if the current product is smaller than the smallest value found so far
                    if (product < smallestValue) {
                        smallestValue = product;
                        item = i;
                    }
                }

                var {x0, y0, x1, y1} = getCurrentViewport(this._viewer);
                if (item && item.ann.x0 >= x0 && item.ann.y0 >= y0 && item.ann.x1 <= x1 && item.ann.y1 <= y1) {
                    console.log('Mouse click on Konva shape and hide:', item.ann.id, item);
                    this._viewer.gestureSettingsMouse.clickToZoom = false;
                    this._modifingNode?.show();
                    this._modifingNode = item;
                    item.hide();

                    let svgAnno = konva2w3c(item)[0];
                    console.log('After convert Konva shape to W3C svg:', svgAnno);
                    this._annotoriousLayer.addAnnotation(svgAnno);
                    this._annotoriousLayer.selectAnnotation(svgAnno);
                } else {
                    this._viewer.gestureSettingsMouse.clickToZoom = true;
                }
            }
        });

        this._annotoriousLayer.on('createSelection', selection => {
            let button = document.getElementById('smartpoly');
            if (button && button.value === 'on') {
                let svgSelector = selection ? selection.target.selector.value : null;
                let query = extractSelectorInfo(svgSelector);

                let api = segmentAPI;
                refineSelection(api, query).then(ann => {
                    let item = parseShape(ann);
                    let newShape = buildSVGTarget(item);
                    selection.target = newShape;

                    let label = document.getElementById('tag-input').value;
                    if (label.trim() != '') {
                        selection.body.push({
                            "type": "TextualBody",
                            "purpose": "tagging",
                            "value": label.trim(),
                        });
                    }

                    this._annotoriousLayer.updateSelected(selection, true);
                    const toolButton = document.querySelector('button.a9s-toolbar-btn.'.concat(query.shape));
                    if (toolButton) {
                        toolButton.click(); // Simulate a click to activate the drawing tool
                    }
                });
            }
        });

        this._annotoriousLayer.on('cancelSelected', annotation => {
            this._modifingNode?.show();
            this._annotoriousLayer.removeAnnotation(annotation);
            this._annotoriousLayer.clearAnnotations();
            this._modifingNode = null;
        });

        // Update konva object with annotorious changes
        this._annotoriousLayer.on('createAnnotation', annotation => {
            let query = w3c2konva(annotation);  // JSON.stringify(annotation)
            query['annotator'] = this.userId.toString();
            query['created_at'] = new Date().toISOString();
            this._annotoriousLayer.removeAnnotation(annotation);
            this._annotoriousLayer.cancelSelected();
            this._annotoriousLayer.clearAnnotations();

            let api = this.APIs.annoInsertAPI;
            createAnnotation(api, query).then(ann => {
                let item = this.createKonvaItem(ann);
                this.getLayerQueue().add(item);
                this.addAnnotator(this.userId);  // display the login user's annotation after creating a new annotation
                // this.activeAnnotators need to filter out model annotator
                const filteredActiveAnnotators = [];
                this.activeAnnotators.forEach(item => {
                    if (!isNaN(item)) {
                        filteredActiveAnnotators.push(item);
                    }
                  });
                drawNUpdateDatatable(this.APIs.annoSearchAPI, {"annotator": filteredActiveAnnotators});
            });
        });

        this._annotoriousLayer.on('updateAnnotation', (annotation, previous) => {
            let query = w3c2konva(annotation);
            query['annotator'] = this.userId.toString();
            query['created_at'] = new Date().toISOString();

            let item = this._modifingNode;
            let api = this.APIs.annoUpdateAPI + item.ann.id;
            updateAnnotation(api, query).then(ann => {
                item = this.updateKonvaItem(ann, item);
                item.show();
                this.addAnnotator(this.userId);
                const filteredActiveAnnotators = [];
                this.activeAnnotators.forEach(item => {
                    if (!isNaN(item)) {
                        filteredActiveAnnotators.push(item);
                    }
                  });
                drawNUpdateDatatable(this.APIs.annoSearchAPI, {"annotator": filteredActiveAnnotators}, this.colorPalette);
            });
            this._annotoriousLayer.removeAnnotation(annotation);
            this._annotoriousLayer.clearAnnotations();
            this._modifingNode = null;
        });

        this._annotoriousLayer.on('deleteAnnotation', annotation => {
            let item = this._modifingNode;
            let api = this.APIs.annoDeleteAPI + item.ann.id;
            deleteAnnotation(api).then(resp => {
                this.getLayerQueue().remove(item);
                const filteredActiveAnnotators = [];
                this.activeAnnotators.forEach(item => {
                    if (!isNaN(item)) {
                        filteredActiveAnnotators.push(item);
                    }
                  });
                drawNUpdateDatatable(this.APIs.annoSearchAPI, {"annotator": filteredActiveAnnotators});
            });
            this._annotoriousLayer.cancelSelected();
            this._annotoriousLayer.removeAnnotation(annotation);
            this._annotoriousLayer.clearAnnotations();
            this._modifingNode = null;
        });
    }

    addLayer(id, capacity=null) {
        let layer = new Konva.Layer();
        this._konvaStage.add(layer);
        this.layerQueues[id] = new LayerFIFOQueue(layer, capacity=capacity);
        console.log("Create new layer: ", capacity, this.layerQueues[id]);
    }

    konvaStage() {
        return this._konvaStage;
    }

    konvaLayers() {
        return this._konvaStage.getLayers();
    }

    getLayerQueue(id='konvaLayer-0') {
        return this.layerQueues[id];
    }

    clear() {
        for (let key in this.layerQueues) {
            this.layerQueues[key].destroyChildren();
        }
        this._konvaStage.clear();
    }

    resize() {
        if (this._containerWidth !== this._viewer.container.clientWidth) {
            this._containerWidth = this._viewer.container.clientWidth;
            this._canvasdiv.setAttribute('width', this._containerWidth);
        }

        if (this._containerHeight !== this._viewer.container.clientHeight) {
            this._containerHeight = this._viewer.container.clientHeight;
            this._canvasdiv.setAttribute('height', this._containerHeight);
        }
    }
    
    resizeCanvas() {
        let origin = new OpenSeadragon.Point(0, 0);
        let viewportZoom = this._viewer.viewport.getZoom(true);
        this._konvaStage.setWidth(this._containerWidth);
        this._konvaStage.setHeight(this._containerHeight);

        let imageSize = this._viewer.world.getItemAt(0).getContentSize();
        let zoom = this._viewer.viewport._containerInnerSize.x * viewportZoom / imageSize.x;
        this._konvaStage.scale({ x: zoom, y: zoom });

        let viewportWindowPoint = this._viewer.viewport.viewportToWindowCoordinates(origin);
        let x = Math.round(viewportWindowPoint.x);
        let y = Math.round(viewportWindowPoint.y);
        let canvasOffset = this._canvasdiv.getBoundingClientRect();

        let pageScroll = OpenSeadragon.getPageScroll();
        this._konvaStage.x(x - canvasOffset.x - window.scrollX);
        this._konvaStage.y(y - canvasOffset.y - window.scrollY);
        this._konvaStage.draw();
    }

    createKonvaItem(ann) {
        let itemCfgs = parseDatabaseAnnotation(ann, this.colorPalette);
        let item;
        if (itemCfgs["shape"] === "polygon") {
            item = new Konva.Line(itemCfgs);
        } else if (itemCfgs["shape"] == "rect") {
            item = new Konva.Rect(itemCfgs);
        } else if (itemCfgs["shape"] == "point") {
            item = new Konva.Star(itemCfgs);
        } else {
            item = new Konva.Ellipse(itemCfgs);
        }
        item.ann = ann;

        return item;
    }

    updateKonvaItem(ann, item) {
        let itemCfgs = parseDatabaseAnnotation(ann, this.colorPalette);
        item.setAttrs(itemCfgs);
        item.ann = ann;

        return item;
    }

    removeAllShapes() {
        let layerQueue = this.getLayerQueue();
        layerQueue.destroyChildren();
    }
}

// export default IViewerAnnotation;