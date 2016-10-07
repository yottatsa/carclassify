window.onload = function () {
    function Model() {
        var self = this;
        var labelHash = {};
        this.labels = ko.observableArray();
        this.images = ko.observableArray();
        this.trash = function (data, model) {
            if (data instanceof Label) {
                model.labels.remove(data);
                data.estimates.forEach(function(estimate) {
                    estimate.delete();
                });
                data.probes().forEach(function(probe) {
                    probe.delete();
                });
                delete labelHash[data.name];
            };
            if (data instanceof Probe) {
                data.delete();
                data.image.updateEstimate(data.label);
                data.label.probes.remove(data);
            };
        };
        this.addLabel = function(name, template) {
            if (template) {
                var label;
                var i=0;
                while (!label || label in labelHash) {
                    i++;
                    label = template + i;
                }
                name = label;
            }
            if (name in labelHash) {
                return labelHash[name]
            } else {
                var label = new Label(name);
                labelHash[name] = label;
                this.labels.push(label);
                return label
            }
        };
    };

    var model = ko.mapping.fromJS(new Model(),
    {
        'images': {
            create: function(options) {
                return new Image(options.data);
            }
        }
    });

    var Label = function(name) {
        this.name = name;
        this.showonly = ko.observable()
        this.estimate = "0%";
        this.probes = ko.observableArray();
        this.estimates = [];
        this.amount = ko.pureComputed(function() {
            return this.probes().length;
        }, this);
    }

    var Estimate = function(label, image) {
        var self = this;
        this.image = image;
        this.label = label
        this.name = label.name
        this.estimate = "0%";
        this.delete = function () {
            self.image.estimates.remove(self);
        };
    }

    var Probe = function(image, label, x, y) {
        var self = this;
        this.image = image
        this.label = label;
        this.name = label.name;
        this.x = ko.observable(x);
        this.y = ko.observable(y);
        this.delete = function () {
            self.image.probes.remove(self);
        };
    }

    var Image = function(data) {
        var self = this;
        var labels = [];
        ko.mapping.fromJS(data,
        {
            'probes': {
                create: function(options) {
                    var label = model.addLabel(options.data.label)
                    if (labels.indexOf(label) == -1) {
                        labels.push(label)
                    }
                    var probe = new Probe(self, label, options.data.x, options.data.y);
                    label.probes.push(probe)
                    return probe
                }
            }
        }, this);
        this.filteredProbes = ko.pureComputed(function() {
            var filter = model.labels().filter(function (item) {
                return item.showonly();
            });
            if (filter.length == 0) return self.probes();
            return self.probes().filter(function (item) {
                return (filter.indexOf(item.label) != -1)
            });
        });
        this.estimate = "0%";
        this.estimates = ko.observableArray();
        this.addProbe = function (label, x, y) {
            var probe = new Probe(self, label, x, y);
            self.probes.push(probe);
            label.probes.push(probe)
            return probe;
        }
        this.updateEstimate = function (label) {
            var found = null;
            self.estimates().forEach(function (estimate) {
                if (estimate.label == label) {
                    found = estimate;
                }
            });
            if (found == null) {
                var estimate = new Estimate(label, self);
                self.estimates.push(estimate);
                label.estimates.push(estimate);
            }
            if (found != null) {
                var count = 0;
                self.probes().forEach(function (probe) {
                    if (probe.label == label) {
                        count++;
                    }
                });
                if (count == 0) {
                    found.delete();
                };
            }
        }

        this.dragProbe = function(data, model) {
            var bodyRect = document.body.getBoundingClientRect();
            var rect = this.element.getBoundingClientRect();
            var x = event.offsetX-rect.left+bodyRect.left;
            var y = event.offsetY-rect.top+bodyRect.top;
            if (data instanceof Label) {
                self.addProbe(data, x, y);
                self.updateEstimate(data);
            }
            if (data instanceof Probe) {
                data.x(x);
                data.y(y);
            }
        };
        labels.forEach(this.updateEstimate)
    }

    ko.applyBindings(model);
    function b() {
        var xhr = new XMLHttpRequest();
        xhr.open('GET', '../api/images.json');
        xhr.responseType = 'json';

        xhr.onload = function() {
            ko.mapping.fromJS(xhr.response, model);
            return
            xhr.response.images.forEach(function(item) {
            });
        };

        xhr.onerror = function() {
          console.log("Booo");
        };

        xhr.send();
    }
    b()
};