from xgen.xgen_run import xgen
from train_script_main import training_main
if __name__ == '__main__':

    json_path = 'args_ai_template.json'
    # json_path = 'args_ai_template_sgpu.json'

    def run(onnx_path, quantized, pruning, output_path, **kwargs):
        import random
        res = {}
        # for simulation
        pr = kwargs['sp_prune_ratios']
        res['output_dir'] = output_path

        res['latency'] = 50

        return res


    xgen(training_main, run, xgen_config_path=json_path, xgen_mode='customization')