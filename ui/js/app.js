ko.mapping.defaultOptions().ignore = ['image', 'delete']

$.put = function(url, data, callback, type) {

  if ( $.isFunction(data) ){
    type = type || callback,
    callback = data,
    data = {}
  }

  return $.ajax({
    url: url,
    type: 'PUT',
    success: callback,
    data: data,
    contentType: type
  });
}

function Model() {
    var self = this;
    this.labelHash = {};
    this.score = ko.observable(0);
    this.labels = ko.observableArray();
    this.images = ko.observableArray();
    this.newImages = ko.observableArray();
    this.trash = function (data, model) {
        if (data instanceof Label) {
            data.delete();
        };
        if (data instanceof Probe) {
            data.delete();
        };
    };
    this.addLabel = function(name, template) {
        if (template) {
            var label;
            var i=0;
            while (!label || label in this.labelHash) {
                i++;
                label = template + i;
            }
            name = label;
        }
        if (name in this.labelHash) {
            return this.labelHash[name]
        } else {
            var label = new Label(name);
            this.labelHash[name] = label;
            this.labels.push(label);
            return label
        }
    };
    this.loadModel = function(model) {
        $.get("/api/images.json", function(data) {
            ko.mapping.fromJS({images: data.images, score: data.metadata.score}, model)
        });
    };
    this.saveModel = function(model, event) {
        $(event.target).button('loading');
        $.put("/api/images.json", ko.mapping.toJSON({images: model.images}), function(data) {
            ko.mapping.fromJS(data.metadata, model)
        }, 'application/json').always(function () {
            $(event.target).button('reset');
        });
    };
    this.getNewImages = function(model, event) {
        $(event.target).button('loading');
        $.get("/api/live.json", function(data) {
            ko.mapping.fromJS({newImages: [data.image], segments: data.segments}, model);
            $("#newImageModal").modal();
        }).always(function () {
            $(event.target).button('reset');
        });
    };
    this.saveNewImages = function(model) {
        ko.toJS(model.newImages).forEach(function (item) {
            model.images.push(new Image(item));
        });
        model.newImages.removeAll();
        $("#newImageModal").modal('hide');
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
    var self = this;
    this.name = name;
    this.showonly = ko.observable()
    this.estimate = "0%";
    this.probes = ko.observableArray();
    this.amount = ko.pureComputed(function() {
        return model.images().map(function (i) {return i.probes().filter(function (p) { return p.label() == self.name}).length }).reduce(function (x,y) {return x+y }, 0);
    }, this);

    this.mergeLabels = function(data, model) {
        if (data != self) {
            data.probes().forEach(function (probe) {
                probe.label(self.name);
                self.probes.push(probe);
            });
            data.probes.removeAll();
            data.delete();
        }
    };
    this.delete = function () {
        self.probes().forEach(function(probe) {
            probe.delete();
        });
        delete model.labelHash[this.name];
        model.labels.remove(self);
    };
}

var Probe = function(image, label, x, y) {
    var self = this;
    this.image = image
    this.label = ko.observable(label);
    this.x = ko.observable(x);
    this.y = ko.observable(y);
    this.delete = function () {
        self.image.probes.remove(self);
    };
}

var Image = function(data) {
    var self = this;
    var labels = [];
    this.probes = ko.observableArray();
    ko.mapping.fromJS(data,
    {
        'probes': {
            create: function(options) {
                var label = model.addLabel(options.data.label)
                if (labels.indexOf(label) == -1) {
                    labels.push(label)
                }
                var probe = new Probe(self, label.name, options.data.x, options.data.y);
                label.probes.push(probe)
                return probe
            }
        }
    }, this);
    this.filteredProbes = ko.pureComputed(function() {
        var filter = model.labels().filter(function (item) {
            return item.showonly();
        }).map(function (item) {
            return item.name;
        });
        if (filter.length == 0) return self.probes();
        return self.probes().filter(function (item) {
            return (filter.indexOf(item.label()) != -1)
        });
    });
    this.estimate = "0%";
    this.addProbe = function (label, x, y) {
        var probe = new Probe(self, label.name, x, y);
        self.probes.push(probe);
        label.probes.push(probe)
        return probe;
    }
    this.dragProbe = function(data, model) {
        var bodyRect = document.body.getBoundingClientRect();
        var rect = this.element.getBoundingClientRect();
        var x = event.offsetX-rect.left+bodyRect.left;
        var y = event.offsetY-rect.top+bodyRect.top;
        if (data instanceof Label) {
            self.addProbe(data, x, y);
        }
        if (data instanceof Probe) {
            data.x(x);
            data.y(y);
        }
    };
}

window.onload = function () {
    ko.applyBindings(model);
    model.loadModel(model);
};