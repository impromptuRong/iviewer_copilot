// const { findLastKey } = require("lodash");

// Color Palette
class ColorPalette {
    constructor(colorPalette={}, defaultColor='#E8E613') {  // '#fd7e14'
        this.colorPalette = colorPalette;
        this.defaultColor = defaultColor;
    }

    addColor(label, color) {
        if (this.colorPalette[label] === undefined) {
            this.colorPalette[label] = color;
        } else {
            throw new Error(`label=${label} exists in Palette. `);
        }
    }

    getColor(label, defaultColor=null) {
        if (this.colorPalette[label] === undefined) {
            let newColor = defaultColor || this.defaultColor;
            this.addColor(label, newColor);
        }

        let color = this.colorPalette[label];
        return {'border': color + 'ff', 'face': color + '35', 
                'box-shadow': `0 0 0 10px ${color}, inset 0 0 0 10px ${color}` }
    }

    labels() {
        return Object.keys(this.colorPalette);
    }
}


function parseDatabaseAnnotation(ann, colorPalette) {
    let color = colorPalette.getColor(ann.label);

    // poly_x and poly_y are strings of comma-separated floats
    let polyX = ann.poly_x.split(',');
    let polyY = ann.poly_y.split(',');

    let itemCfgs;
    if (ann.poly_x.length === 0 || ann.poly_y.length === 0) {  // box
        return {
            id: ann.id,
            label: ann.label,
            description: ann.description,
            annotator: ann.annotator,
            fill: color['face'],
            stroke: color['border'],
            strokeWidth: 2,
            draggable: false,
            shape: "rect",
            x: parseFloat(ann.x0),
            y: parseFloat(ann.y0),
            width: parseFloat(ann.x1) - parseFloat(ann.x0),
            height: parseFloat(ann.y1) - parseFloat(ann.y0),
        };
    } else if (polyX.length === 1 || polyY.length === 1) {  // Ellipse and circle
        let itemShape = parseFloat(polyX[0]) === parseFloat(polyY[0]) ? "circle" : "ellipse";
        return {
            id: ann.id,
            label: ann.label,
            description: ann.description,
            annotator: ann.annotator,
            fill: color['face'],
            stroke: color['border'],
            strokeWidth: 2,
            draggable: false,
            shape: itemShape,
            x: parseFloat(ann.xc),
            y: parseFloat(ann.yc),
            radiusX: parseFloat(polyX[0]),
            radiusY: parseFloat(polyY[0]),
        };
    } else {  // Polygon
        let points = [];
        for (let i = 0; i < polyX.length && i < polyY.length; i++) {
            points.push(parseFloat(polyX[i]));
            points.push(parseFloat(polyY[i]));
        }
        return {
            id: ann.id,
            label: ann.label,
            description: ann.description,
            annotator: ann.annotator,
            fill: color['face'],
            stroke: color['border'],
            strokeWidth: 2,
            draggable: false,
            shape: "polygon",
            points: points,
            closed: true,
        };
    }
}


function konva2w3c(selectedShape) {
    // konva shapes: rect, ellipse, circle, polygon
    const { id, label, annotator, shape, description} = selectedShape.attrs;
    const bbox = { x0: selectedShape.ann.x0, y0: selectedShape.ann.y0, x1: selectedShape.ann.x1, y1: selectedShape.ann.y1 };

    let value, type;
    let rectflag = false;
    let labelArray = label.split(',');
    let taggingResult = [];
    if (labelArray.length > 0 && labelArray[0] !== "") {
        for (let item of labelArray) {
            taggingResult.push({
                "type": "TextualBody",
                "value": item.trim(),
                "purpose": "tagging"
            });
        }
    } 
    if (shape === "polygon") {
        const {points} = selectedShape.attrs;
        const pointsString = points.reduce((acc, val, index, array) => {
            if (index % 2 === 0) {
                acc += val + ",";
            } else {
                acc += val + " ";
            }
 
            if (index === array.length - 1) {
                acc = acc.trim();
            }
 
            return acc;
        }, "");
 
        value = `<svg><polygon points="${pointsString}"></polygon></svg>`;
        type = "SvgSelector";
    } else if (shape === "ellipse") {
        const {x, y, radiusX, radiusY} = selectedShape.attrs;
        type = "SvgSelector";
        value = `<svg><ellipse cx="${x}" cy="${y}" rx="${radiusX}" ry="${radiusY}"></ellipse></svg>`;
    } else if (shape === "circle") {
        const {x, y, radiusX, radiusY} = selectedShape.attrs;
        type = "SvgSelector";
        value = `<svg><circle cx="${x}" cy="${y}" r="${radiusX}"></circle></svg>`;
    } else {
        rectflag = true;
        const {x, y, width, height} = selectedShape.attrs;
        type = "FragmentSelector";
        value = `xywh=pixel:${x},${y},${width},${height}`
    }
 
    const w3cAnnotation = [
        {
            "type": "Annotation",
            "body": [
                {
                    "type": "TextualBody",
                    "value": description || '',
                    "purpose": "commenting"
                },
                // {
                //     "type": "TextualBody",
                //     "value": '',
                //     "purpose": "replying"
                // },
                ...taggingResult,
                // {
                //     "type" : "",
                //     "value" : bbox,
                //     "purpose" : "aiAssistant"
                // },
                {
                    "type" : "annotator",
                    "value" : annotator,
                    "purpose": "showannotator"
                },
            ],
            "target": {
                "selector": {
                    "type": type,
                    ...(rectflag ? {"conformsTo": "http://www.w3.org/TR/media-frags/"} : {}),
                    "value": value
                }
            },
            "id": `#${id}`
        }
    ];

    return w3cAnnotation;
}


function w3c2konva(w3cAnnotation) {
    // Extract data from the provided W3C annotation format
    const svgSelector = w3cAnnotation ? w3cAnnotation.target.selector.value : null;
    let description = "";
    let label = "";
    var labels = [];
    w3cAnnotation.body.find(function(b) {
        if (b.purpose == 'tagging') {
            labels.push(b.value);
        }
    })
    label = labels.join(',');

    var currentDes = w3cAnnotation ? w3cAnnotation.body.filter(function(b) {
        return b.purpose == 'commenting';
    }) : [];
    var currentReplies = w3cAnnotation ? w3cAnnotation.body.filter(function(b) {
        return b.purpose == 'replying';
    }) : [];
    
    description = '';
    if (currentDes && currentDes.length > 0) {
        description += currentDes.map(function(item) {
            console.log(item.value)
            return item.value;
        }).join('\n');
    }
    if (currentDes.length > 0 && currentReplies.length > 0){
        description += '\n';
    }
    if (currentReplies.length > 0) {
        description += currentReplies.map(function(reply) {
            return reply.value;
        }).join('\n');
    }

    // if(w3cAnnotation.body.length>0){
    //     description = w3cAnnotation.body[0].value;
    // }
    console.log("selector", svgSelector);
    // svg shapes: rect, ellipse, circle, polygon, path
    let convertedObject = {};

    // Parse the SVG selector to extract polygon points
    if (svgSelector.startsWith('<svg><polygon')) {
        const match = /points="([^"]+)"/.exec(svgSelector);
        const pointsString = match ? match[1] : '';
        console.log("svg", pointsString);
        // Split the points string and convert to array
        const pointPairs = pointsString.split(' ').flatMap(pair => pair.split(',').map(Number));
        console.log("pairs", pointPairs);
        let xArray = [];
        let yArray = [];
        // Extract x and y arrays without rounding
        for (let i = 0; i < pointPairs.length; i++) {
            if (i % 2 === 0) {
                xArray.push(pointPairs[i]);
            } else {
                yArray.push(pointPairs[i]);
            }
        }
        // const xArray = pointPairs.map(point => parseFloat(point[0]));
        console.log("x",xArray)
        // const yArray = pointPairs.map(point => parseFloat(point[1]));
        console.log("y",yArray)
        // Find minX, maxX, minY, maxY as integers
        const minX = Math.floor(Math.min(...xArray));
        const maxX = Math.ceil(Math.max(...xArray));
        const minY = Math.floor(Math.min(...yArray));
        const maxY = Math.ceil(Math.max(...yArray));

        // Create poly_x and poly_y strings with two decimal places
        const polyX = xArray.map(x => (parseFloat(x).toFixed(2)).toString()).join(',');
        const polyY = yArray.map(x => (parseFloat(x).toFixed(2)).toString()).join(',');
        // console.log("polyx", polyX)
        // console.log("polyx", polyY)
        // Create an object with the extracted data
        convertedObject = {
            // id: id,
            poly_x: polyX,
            poly_y: polyY,
            description: description,
            x0: minX,
            y0: minY,
            x1: maxX,
            y1: maxY,
            label: label,
//             annotator: userid.toString(),
            shape:"polygon"
        };
    } else if (svgSelector.startsWith('<svg><ellipse')) {
        const ellipseMatch = svgSelector.match(/<ellipse cx="(\d+(\.\d+)?)".* cy="(\d+(\.\d+)?)".* rx="(\d+(\.\d+)?)".* ry="(\d+(\.\d+)?)".*<\/ellipse>/);
        if (ellipseMatch) {
            const [, cx, , cy, , rx, , ry] = ellipseMatch;
            const parsedXC = parseFloat(cx);
            const parsedYC = parseFloat(cy);
            const parsedRX = parseFloat(rx);
            const parsedRY = parseFloat(ry);
            return {
                label: label,
//                 annotator: userid.toString() ,
                description: description,
                poly_x: parsedRX.toFixed(2),
                poly_y: parsedRY.toFixed(2),
                x0: (parsedXC - parsedRX).toFixed(2),
                y0: (parsedYC - parsedRY).toFixed(2),
                x1: (parsedXC + parsedRX).toFixed(2),
                y1: (parsedYC + parsedRY).toFixed(2),
                xc: parsedXC.toFixed(2),
                yc: parsedYC.toFixed(2),
                shape: "ellipse"
            };
        }
    } else if (svgSelector.startsWith('<svg><circle')) {
        // Circle format: <circle cx="x" cy="y" r="radius"></circle>
        const circleMatch = svgSelector.match(/<circle cx="(\d+(\.\d+)?)".* cy="(\d+(\.\d+)?)".* r="(\d+(\.\d+)?)".*<\/circle>/);
        if (circleMatch) {
            const [, cx, , cy, , r] = circleMatch;
            const parsedXC = parseFloat(cx);
            const parsedYC = parseFloat(cy);
            const parsedR = parseFloat(r);
            return {
                label: label,
//                 annotator: userid.toString(),
                description: description,
                poly_x: parseFloat(r).toFixed(2),
                poly_y: parseFloat(r).toFixed(2),
                x0: (parsedXC - parsedR).toFixed(2),
                y0: (parsedYC - parsedR).toFixed(2),
                x1: (parsedXC + parsedR).toFixed(2),
                y1: (parsedYC + parsedR).toFixed(2),
                xc: parseFloat(cx).toFixed(2),
                yc: parseFloat(cy).toFixed(2),
                shape: "circle"
            };
        }
    } else if (svgSelector.startsWith('xywh=pixel:')) {
        const rectMatch = svgSelector.match(/xywh=pixel:(\d+(\.\d+)?),(\d+(\.\d+)?),(\d+(\.\d+)?),(\d+(\.\d+)?)/);
        if (rectMatch) {
            const [, x, , y, , width, , height] = rectMatch;
              // Parse values to numbers
            const parsedX = parseFloat(x);
            const parsedY = parseFloat(y);
            const parsedWidth = parseFloat(width);
            const parsedHeight = parseFloat(height);
            return {
                label: label,
//                 annotator: userid.toString(),
                description: description,
                poly_x: "",
                poly_y: "",
                x0: parsedX.toFixed(2),
                y0: parsedY.toFixed(2),
                x1: (parsedX + parsedWidth).toFixed(2),
                y1: (parsedY + parsedHeight).toFixed(2),
                shape: "rect"
            };
        }
    } else if (svgSelector.startsWith('<svg><path')) {
        const pathMatch = svgSelector.match(/<path d="([^"]*)".*<\/path>/);
        if (pathMatch) {
            const pathData = pathMatch[1];
            // Parse the path data into an array of commands
            const pathCommands = pathData.split(/(?=[MLCZ])/).filter(Boolean);

            // Extract x and y coordinates from path commands
            const polyX = [];
            const polyY = [];

            pathCommands.forEach(command => {
                const params = command.substring(1).split(' ').map(parseFloat);
                polyX.push(parseFloat(params[0]).toFixed(2));
                polyY.push(parseFloat(params[1]).toFixed(2));
            });
            const minX = Math.floor(Math.min(...polyX));
            const maxX = Math.ceil(Math.max(...polyX));
            const minY = Math.floor(Math.min(...polyY));
            const maxY = Math.ceil(Math.max(...polyY));

            return {
                label: label,
//                 annotator: userid.toString(),
                description: description,
                poly_x: polyX.join(','),
                poly_y: polyY.join(','),
                x0: minX,
                y0: minY,
                x1: maxX,
                y1: maxY,
                shape: "freehand"
            };
        }
    }

    return convertedObject;
}


function createAnnotation(api, query) {
    console.log("createAnnotation", api, query);
    return fetch(api, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(query)
    }).then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    }).catch(error => {
        console.error('There was a problem with the fetch operation:', error);
        throw error;
    });
}


function readAnnotation(api) {
    console.log("readAnnotation", api);
    return fetch(api, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        },
    }).then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    }).catch(error => {
        console.error('There was a problem with the fetch operation:', error);
        throw error;
    });
}


function updateAnnotation(api, query) {
    console.log("updateAnnotation", api, query);
    return fetch(api, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(query)
    }).then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    }).catch(error => {
        console.error('There was a problem with the fetch operation:', error);
        throw error;
    });
}


function deleteAnnotation(api) {
    console.log("deleteAnnotation", api);
    return fetch(api, {
        method: 'DELETE',
        headers: {
            'Content-Type': 'application/json'
        },
    }).then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.text();
    }).catch(error => {
        console.error('There was a problem with the delete operation:', error);
        throw error;
    });
}


function searchAnnotations(api, query) {
    console.log("searchAnnotations", api, query);
    return fetch(api, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(query)
    }).then(response => {
        return response.ok ? response.json() : null;
    }).catch(error => {
        console.error('There was a problem with the fetch operation:', error);
        throw error;
    });
}


function countAnnotations(api){
    return fetch(api, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(query)
    }).then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    }).catch(error => {
        console.error('There was a problem with the fetch operation:', error);
        throw error;
    });
}


function getAnnotators(api) {
    console.log("getAnnotators", api);
    return fetch(api, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        },
    }).then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    }).catch(error => {
        console.error('There was a problem with the fetch operation:', error);
        throw error;
    });
}


function getAIDescription(api, query) {
    console.log("aiGeneration", api, query);
    let queryString = Object.keys(query).map(key => `${encodeURIComponent(key)}=${encodeURIComponent(query[key])}`).join('&');
    let url = `${api}&${queryString}`;

    return fetch(url, {
        method: 'GET',
    }).then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.text();
    }).catch(error => {
        console.error('There was a problem with the fetch operation:', error);
        throw error;
    });
}


function getCurrentViewport(viewer) {
    // Get the current viewport rectangle
    if (!viewer.world.getItemCount()) {
        return null;
    }
    let slideImage = viewer.world.getItemAt(0);
    let viewportRect = viewer.viewport.getBounds(true);
    let imageTopLeft = slideImage.viewportToImageCoordinates(
        viewportRect.x, viewportRect.y
    );
    let imageBottomRight = slideImage.viewportToImageCoordinates(
        viewportRect.x + viewportRect.width, viewportRect.y + viewportRect.height
    );
    let bbox = {
        'x0': imageTopLeft.x,
        'y0': imageTopLeft.y,
        'x1': imageBottomRight.x,
        'y1': imageBottomRight.y,
    };
    // console.log('Bbox:', bbox, this._viewer.viewport.getZoom(), viewer.viewport.getCenter());
    return bbox;
}


//*****************************************************//
// The following functions are maintained by Danni

addTile = (viewer, regionalTile) => {
    console.log("reginaltile", regionalTile)
    viewer.addTiledImage({
        tileSource: regionalTile,
        x: 0,
        opacity: 0.5
    });
}


//stop making requests
removeTile = (viewer, tileName) => {
    var count = viewer.world.getItemCount();
    for (i = 0; i < count; i++) {
        tiledImage = viewer.world.getItemAt(i);
        if (tiledImage.source.queryParams.input === tileName) {
            console.log("removetile", tiledImage.source.queryParams.input)
            //set selected addedtileimage opacity to 0
            tiledImage.setOpacity(0);
            // viewer.world.removeItem(tiledImage);
            break;
        }
    }
}


function annotatorTreeBtnClick(annotationLayer, userIdNameMap, treeDomId, floatingWindowDomId, colorPalette){
    let api = annotationLayer.APIs.annoGetAnnotators;
    getAnnotators(api).then(annotatorIds => {
        let annotatorsMap = {};
        annotatorIds.forEach(key => {
            annotatorsMap[key] = userIdNameMap[key];
        });
        if (Object.keys(annotatorsMap).length === 0) {
            $(treeDomId).html('No one or model has annotated this image yet. Click the model button above to start automatic annotation, or use the manual annotation tool to annotate manually.');
        } else {
            $(treeDomId).html("")
            populateJStree(annotationLayer, annotatorsMap, treeDomId, colorPalette);
        }
        
    });
    // dragElement(document.getElementById(floatingWindowDomId));
    // $(floatingWindowDomId).dialog();
    // $(floatingWindowDomId).fadeIn();
}


function populateJStree(annotationLayer, annotatorsMap, treeDomId, colorPalette){
    const treeData = convertToJstreeData(annotatorsMap);
    if ($(treeDomId).jstree(true)) {
        $(treeDomId).jstree('destroy');
    }
    // else {
    //      $(treeDomId).jstree(true).settings.core.data = treeData;
    //      $(treeDomId).jstree(true).refresh();
    //      console.log("update tree data")
         //check annotators which are in activeAnnotators set
        //  let activeIds = annotationLayer.activeAnnotators;
        //  console.log("activeids", activeIds)
        //  activeIds.forEach(function(value) {
        //     $(treeDomId).off("changed.jstree")
        //     $(treeDomId).jstree(true).select_node(value);
        //     $(treeDomId).jstree(true).set_icon(value, "fas fa-eye");
        //     // $(treeDomId).jstree("select_node", value, true);
        // });
        // $(treeDomId).on("changed.jstree")
    // }
    $(treeDomId).jstree({
        "plugins": ["checkbox", "types"],
        "types": {
            "default": {
                "icon": "",
            },
            "file": {
                "icon": "fa fa-eye",
            }
        },
        "core": {
            "data": treeData,
        },
        "state": {
            "opened": ["all"],
        }
    });
    $(treeDomId).on("ready.jstree", function (e, data) {
        let activeIds = annotationLayer.activeAnnotators;
        console.log("activeids", activeIds);
        $(treeDomId).off("changed.jstree") //remove changed.jstree listener before checking previously checked node
        activeIds.forEach(function(value) {
            $(treeDomId).jstree(true).select_node(value);
            $(treeDomId).jstree(true).set_icon(value, "fas fa-eye");
        });
        handCheckboxChange(annotationLayer, treeDomId, annotatorsMap, colorPalette); //add back listener
    });
}


function handCheckboxChange(annotationLayer, treeDomId, childrenNodeMap, colorPalette) {
    //js tree check event
    var checkboxCooldown = false;
    $(treeDomId).on("changed.jstree", function (e, data) {
        var tablequery = {"annotator": []};    
        //control users' clicking speed
        if (!checkboxCooldown) {
            checkboxCooldown = true;
            // Disable all checkboxes
            $('a.jstree-anchor').addClass('disabled-checkbox');
            setTimeout(function () {
                checkboxCooldown = false; // Reset cooldown flag after 1 second
                $('a.jstree-anchor').removeClass('disabled-checkbox');
                // var allNodes = $(treeDomId).jstree(true).get_json('#', { flat: true });
                var checkedNodes = data.instance.get_checked(true);
                // var checkedNodeIds = checkedNodes.map(node => node.id);
                var checkedChildrenNode = checkedNodes.filter(node => node.children.length == 0);   
                //make all children node icon to fa-eye-slash             
                Object.keys(childrenNodeMap).forEach(function(childNodeId) {
                    // Set icon for each child node
                    $(treeDomId).jstree(true).set_icon(childNodeId, "fas fa-eye-slash");
                });
                //update active annotators 
                var updatedNodeid = new Set();
                checkedChildrenNode.forEach(function (node) {
                    var currentIcon = $(treeDomId).jstree(true).get_icon(node.id);
                    if(currentIcon === "fas fa-eye-slash"){
                        $(treeDomId).jstree(true).set_icon(node.id, "fas fa-eye");
                    }
                    updatedNodeid.add(node.id);
                });
                //used to draw history annotation table (only children node under manual can be displayed in the table)
                checkedNodes.forEach(function (node) {
                    if (node.parent === 'manual' && node.children.length === 0) {
                        tablequery.annotator.push(node.id);
                    }
                });
                if (treeDomId === "#layers-left") {
                    drawNUpdateDatatable(annotationLayer.APIs.annoSearchAPI, tablequery, colorPalette);   
                }
                annotationLayer.updateAnnotators(updatedNodeid);
            }, 1000);

            //select the last children node will also select parent node
            // checkedNode = checkedNodes.filter(node => node.children.length ==0 );
            // uncheckedNodes = allNodes.filter(node => !checkedNodeIds.includes(node.id) && node.parent !== "#");
            // // var removedNodes = previousCheckedNodes.filter(node => !checkedNodes.includes(node)).filter(node => node.children.length ==0);
            
            // checkedNode.forEach(function(node, index){
            //     $(treeDomId).jstree(true).set_icon(node.id, "fas fa-eye");
            //     annotationLayer.addAnnotator(node.id);
            //     console.log("add node id", node.id)
            //     // setTimeout(function () {
            //     //     $(treeDomId).jstree(true).set_icon(node.id, "fas fa-eye");
            //     //     annotationLayer.addAnnotator(node.id);
            //     //     console.log("add node id", node.id);
            //     // }, index * 2000);
            // });
            // uncheckedNodes.forEach(function(node){
            //     $(treeDomId).jstree(true).set_icon(node.id, "fas fa-eye-slash");
            //     annotationLayer.removeAnnotator(node.id);
            //     console.log("remove node id", node.id)
            // });
        }
    });
}


function convertToJstreeData(data) {
    var convertedData = [];

    // Create parent nodes
    var modelNode = {
        'id': 'model',
        'text': 'Model Annotation Display',
        'icon': 'fas fa-robot',
        'children': []
    };

    var manualNode = {
        'id': 'manual',
        'text': 'Manual Annotation Display',
        'icon': 'fas fa-user',
        'children': []
    };

    // Iterate over the input data object
    for (var key in data) {
        // Skip null and undefined values
        if (data[key] === null || typeof data[key] === 'undefined') {
            continue;
        }

        // Determine parent node based on key type
        var parentNode = null;
        if (!isNaN(key)) { // Check if key is a number
            parentNode = manualNode;
        } else if (typeof key === 'string') { // Check if key is a string
            parentNode = modelNode;
        }

        // Add a new node to the parent node's children array
        if (parentNode) {
            parentNode.children.push({
                'id': key,
                'text': data[key],
                'annotator': data[key],
                "icon": "fas fa-eye-slash"
            });
        }
    }

    // Add parent nodes to the converted data array
    if (modelNode.children.length > 0) {
        convertedData.push(modelNode);
    }
    if (manualNode.children.length > 0) {
        convertedData.push(manualNode);
    }

    return convertedData;
}


// Function to find the path by id
function getModelApiById(array, id) {
    for (var i = 0; i < array.length; i++) {
        if (array[i].id === id) {
            return array[i].path; // Return the path if id matches
        }
    }
    return null; // Return null if id is not found
}


function createOverlayElement(viewer) {
    var overlay = $('<div>').css({
        position: 'absolute',
        top: '0',
        right: '0',
        backgroundColor: '#fdf3d8',
        color: '#806520',
        padding: '0.75rem 1.25rem',
        borderRadius: '0.25rem',
        border: '1px solid #fceec9',
        zIndex: '9999',
        display: 'none' // Initially hide the overlay
    }).appendTo(viewer.container);
    
    return overlay;
}


function showInstructions(overlayElement, message) {
    // Display overlay with instructions
    if (!overlayElement.is(":visible")) {
        overlayElement.text(message).fadeIn();
    }
}


function hideInstructions(overlayElement) {
    // Hide overlay
    if (overlayElement.is(":visible")) {
        overlayElement.fadeOut();
    }
}


function drawNUpdateDatatable(api, query, colorPalette){
    if (query.annotator.length > 0) { //only draw rows with selected manual annotator in the table
        searchAnnotations(api, query).then(ann => {
            if ($.fn.DataTable.isDataTable('#dataTable')) {
                $('#dataTable').DataTable().clear().destroy();
            }
            if (ann) { // if return is not null
                ann.reverse();
                var data = ann.map(annotation => [
                    userIdNameMap[annotation.annotator],
                    annotation.description,
                    annotation.label,
                    `[${annotation.x0}, ${annotation.x1}, ${annotation.y0}, ${annotation.y1}]`,
                    annotation
                ]);
                
                $('#dataTable').DataTable({
                    data: data,
                    scrollX: true,
                    columns: [
                        { title: 'Annotator' },
                        { title: 'Description' },
                        { title: 'Label' },
                        { title: 'Coordinates' },
                        { title: 'Zoom to Annotation', orderable: false } // New column for actions
                    ],
                    columnDefs: [{
                        targets: -1, 
                        render: function (data, type, row, meta) {
                            return '<i class="fas fa-search zoom-to-annotation"></i>';
                        }
                    }],
                    dom: 'Bfrtip',
                    buttons: [
                        {
                            text: '<i class="fa fa-download" style="margin-right: 5px"></i>Download as CSV',
                            className: 'btn-sm btn-dt',
                            action: function (e, dt, button, config) {
                                downloadAllAsCSV(ann); // Call function to download all annotations
                            }
                        }, {
                            text: '<i class="fas fa-sync-alt" style="margin-right: 5px"></i>Reload Table',
                            className: 'btn-sm btn-dt',
                            action: function (e, dt, button, config) {
                                drawNUpdateDatatable(api, query, colorPalette); // Call function to download all annotations
                            }
                        }
                    ]
                });

                $('#dataTable tbody').on('click', 'td:nth-child(5)', function () {
                    var coordinateStr = $(this).closest('tr').find('td:nth-child(4)').text();// Get the text content of the clicked cell
                    var coordinates = coordinateStr.substring(1, coordinateStr.length - 1).split(', '); // Parse coordinates from string
                    var x0 = parseFloat(coordinates[0]);
                    var x1 = parseFloat(coordinates[1]);
                    var y0 = parseFloat(coordinates[2]);
                    var y1 = parseFloat(coordinates[3]);

                    var viewportTopLeft = leftViewer.viewport.imageToViewportCoordinates(
                        new OpenSeadragon.Point(x0, y0),
                    );

                    var viewportBottomRight = leftViewer.viewport.imageToViewportCoordinates(
                        new OpenSeadragon.Point(x1, y1)
                    );

                    var bounds = new OpenSeadragon.Rect(
                        viewportTopLeft.x,
                        viewportTopLeft.y,
                        viewportBottomRight.x - viewportTopLeft.x,
                        viewportBottomRight.y - viewportTopLeft.y
                    );
                    leftViewer.viewport.fitBounds(bounds, true); 
                    leftViewer.viewport.zoomBy(0.9);
                });
            }
        });
    } else {
        if ($.fn.DataTable.isDataTable('#dataTable')) {
            $('#dataTable').DataTable().clear().destroy();
        }
    }
}


function downloadAllAsCSV(annotations) {
    // Convert annotations to CSV format
    var headerRow = ['Annotator', 'Description', 'Label', 
                     'x0', 'y0', 'x1', 'y1', 'xc', 'yc',
                     'poly_x', 'poly_y'];

    var csvData = annotations.map(function (annotation) {
        return [
            userIdNameMap[annotation.annotator], 
            annotation.description, 
            annotation.label, 
            annotation.x0, 
            annotation.y0, 
            annotation.x1, 
            annotation.y1,
            annotation.xc, 
            annotation.yc,
            annotation.poly_x,
            annotation.poly_y
        ].map(escapeCSV).join(',');
    });
//     var headerString = '"' + headerRow.join('","') + '"';
//     var csvDataString = csvData.join('\n');
//     var csvString = headerString + '\n' + csvDataString;
       
    var csvString = ['"' + headerRow.join('","') + '"'].concat(csvData).join('\n');
    // Create a Blob containing the CSV data
    var blob = new Blob([csvString], { type: 'text/csv' });

    // Create a temporary URL for the Blob
    var url = window.URL.createObjectURL(blob);

    // Create a link element and trigger the download
    var a = document.createElement('a');
    a.href = url;
    a.download = 'annotations.csv';
    document.body.appendChild(a);
    a.click();

    // Clean up
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}


function escapeCSV(text) {
    // If the text contains double quotes, escape them by doubling them
    text = String(text);
    if (text.includes('"')) {
        text = text.replace(/"/g, '""');
    }
    text = `"${text}"`;

    return text;
}


function sendMessageWithRetry(webSocket, query) {
    const MAX_RETRIES = 3; 
    const RETRY_INTERVAL = 3000; 
    let retries = 0;
    function retrySend() {
        if (retries < MAX_RETRIES) {
            try {
                webSocket.send(JSON.stringify(query));

            } catch (error) {
                console.log('Error sending message:', error);
                retries++;
                setTimeout(retrySend, RETRY_INTERVAL);
            }
        } else {
            console.error('Maximum retry attempts reached. Message sending failed.');
        }
    }

    retrySend();
}


function freezeAndLoad(editor) {
    // Create a dark layer element
    var darkLayer = document.createElement('div');
    darkLayer.classList.add('dark-layer');
    darkLayer.style.width = editor.offsetWidth + 'px';
    darkLayer.style.height = editor.offsetHeight + 'px';
    editor.appendChild(darkLayer);

    // Add CSS styles 
    darkLayer.style.position = 'absolute';
    darkLayer.style.top = '0px';
    darkLayer.style.left = '0px';
    darkLayer.style.zIndex = '1000';
    darkLayer.style.background = 'rgba(0, 0, 0, 0.5)'; // Semi-transparent black background

    // Create a container for the awesome icon
    var iconContainer = createIconContainer();
    iconContainer.style.position = 'absolute';
    iconContainer.style.top = '50%';
    iconContainer.style.left = '50%';
    iconContainer.style.transform = 'translate(-50%, -50%)';

    // Append the icon container to the dark layer
    darkLayer.appendChild(iconContainer);
    function updateEllipsis() {
        // Update ellipsis content
        var dots = '';
        for (var i = 0; i < numDots; i++) {
            dots += '.';
        }
        ellipsisContainer.textContent = dots;

        // Increase dot count, reset if greater than 3
        numDots = (numDots % 3) + 1;
    } 
    var ellipsisContainer = document.querySelector('.ellipsis');
    var numDots = 1;
    setInterval(updateEllipsis, 500);

    // editor.removeChild(darkLayer);

    return darkLayer;
}


function createIconContainer() {
    var iconContainer = document.createElement('div');
    iconContainer.classList.add('icon-container');

    // Add ellipsis span and chat icon
    var ellipsisSpan = document.createElement('span');
    ellipsisSpan.classList.add('ellipsis');
    ellipsisSpan.textContent = '.';
    iconContainer.appendChild(ellipsisSpan);

    // var chatIcon = document.createElement('i');
    // chatIcon.classList.add('fas', 'fa-comment');
    // iconContainer.appendChild(chatIcon);

    return iconContainer;
}

/**********************************************/
