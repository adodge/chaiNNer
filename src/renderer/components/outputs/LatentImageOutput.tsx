import { NamedExpression } from '@chainner/navi';
import { memo, useEffect, useMemo } from 'react';
import { useContext, useContextSelector } from 'use-context-selector';
import { isStartingNode } from '../../../common/util';
import { BackendContext } from '../../contexts/BackendContext';
import { GlobalContext, GlobalVolatileContext } from '../../contexts/GlobalNodeState';
import { ModelDataTags } from './elements/ModelDataTags';
import { OutputProps } from './props';

export const LatentImageOutput = memo(
    ({ id, outputId, useOutputData, animated, schemaId }: OutputProps) => {
        const type = useContextSelector(GlobalVolatileContext, (c) =>
            c.typeState.functions.get(id)?.outputs.get(outputId)
        );

        const { current } = useOutputData(outputId);

        const { setManualOutputType } = useContext(GlobalContext);
        const { schemata } = useContext(BackendContext);

        const schema = schemata.get(schemaId);

        useEffect(() => {
            if (isStartingNode(schema)) {
                if (current) {
                    setManualOutputType(id, outputId, new NamedExpression('LatentImage'));
                } else {
                    setManualOutputType(id, outputId, undefined);
                }
            }
        }, [id, schemaId, current, outputId, schema, setManualOutputType]);

        const tags = useMemo(() => {
            if (!current) return undefined;

            return [];
        }, [current]);

        return (
            <ModelDataTags
                loading={animated}
                tags={tags}
            />
        );
    }
);
