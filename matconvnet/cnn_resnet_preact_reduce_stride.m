function net = cnn_resnet_preact_reduce_stride(varargin)

opts.Nclass=10; %number of classses (CIFAR-10 / CIFAR-100)
opts = vl_argparse(opts, varargin) ;

net = load('/scratch/shared/nfs1/srebuffi/MCN/imagenet-resnet-50-dag.mat') ;
net = dagnn.DagNN.loadobj(net) ;

% Reduce the stride of conv1
net.layers(1).block.stride = [1 1];

% Freeze layers of the network which are not BN layers
for l = 1:numel(net.layers)
    if ~isa(net.layers(l).block, 'dagnn.BatchNorm') & ~isempty(net.layers(l).params)
        for l1 = 1:numel(net.layers(l).params)
            k = net.getParamIndex(net.layers(l).params{l1}) ;
            net.params(k).learningRate = 0.0 ;
        end
    end
end

% Delete the first maxpool layer
net.removeLayer('fc1000');
net.removeLayer('prob');
net.removeLayer('pool5');
layer = net.layers(net.getLayerIndex('pool1')) ;
net.removeLayer('pool1') ;
net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;


params = net.layers(end).block.initParams() ;
params = cellfun(@gather, params, 'UniformOutput', false) ;
p = net.getParamIndex(net.layers(end).params) ;
[net.params(p).value] = deal(params{:}) ;

% Add the parallel residual adapters
for l = 1:numel(net.layers)
    if isa(net.layers(l).block, 'dagnn.Conv')
        if net.layers(l).block.size(1) == 3
             net.layers(l).block
             net.addLayer([net.layers(l).name '_bis'], ...
               dagnn.Conv('size', [1 1 net.layers(l).block.size(3) net.layers(l).block.size(4)], ...
                          'stride', net.layers(l).block.stride(1), ....
                          'pad', 0, ...
                          'hasBias', false), ...
               net.layers(l).inputs, ...
               {[net.layers(l).name '_bis']}, ...
               [net.layers(l).name '_bis' '_f']) ;
             
             % Initialize params
             params = net.layers(end).block.initParams() ;
             params = cellfun(@gather, params, 'UniformOutput', false) ;
             p = net.getParamIndex(net.layers(end).params) ;
             [net.params(p).value] = deal(params{:}) ;
             
             net.addLayer([net.layers(l).name '_sum'] , ...
                 dagnn.Sum(), ...
                 {[net.layers(l).name '_bis'],net.layers(l).name}, ...
                 [net.layers(l).name '_sum']) ;
             layers = {} ;
             for l2 = 1:numel(net.layers)
                if strcmp(net.layers(l2).inputs{1}, net.layers(l).name)
                    % net.renameVar(net.layers(l2).inputs{1}, [net.layers(l).name '_sum'], 'quiet', true) ;
                    net.setLayerInputs(net.layers(l2).name,{[net.layers(l).name '_sum']});
                end
             end
        end
    end
end

net.addLayer('prediction_avg' , ...
             dagnn.Pooling('poolSize', [4 4], 'method', 'avg'), ...
             'res5cx', ...
             'prediction_avg') ;

net.addLayer('prediction' , ...
             dagnn.Conv('size', [1 1 2048 opts.Nclass]), ...
             'prediction_avg', ...
             'prediction', ...
             {'prediction_f', 'prediction_b'}) ;
params = net.layers(end).block.initParams() ;
params = cellfun(@gather, params, 'UniformOutput', false) ;
p = net.getParamIndex(net.layers(end).params) ;
[net.params(p).value] = deal(params{:}) ;
%Modification from Andrea (similar to imagenet)
f = net.getParamIndex(net.layers(end).params(1)) ;
net.params(f).value = net.params(f).value /10;

net.addLayer('loss', ...
             dagnn.Loss('loss', 'softmaxlog') ,...
             {'prediction', 'label'}, ...
             'objective') ;

net.addLayer('top1error', ...
             dagnn.Loss('loss', 'classerror'), ...
             {'prediction', 'label'}, ...
             'error') ;


%Meta parameters
net.meta.inputSize = [32 32 3] ;
net.meta.trainOpts.learningRate = [0.01*ones(1,2) 0.1*ones(1,80) 0.01*ones(1,40) 0.001*ones(1,40)] ;
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.batchSize = 128 ;
net.meta.trainOpts.momentum = 0.9 ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

end
