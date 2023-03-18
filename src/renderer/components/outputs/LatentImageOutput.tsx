import { literal } from '@chainner/navi';
import { Center, Flex, Spacer, Text } from '@chakra-ui/react';
import { memo, useEffect } from 'react';
import { useContext, useContextSelector } from 'use-context-selector';
import { struct } from '../../../common/types/util';
import { BackendContext } from '../../contexts/BackendContext';
import { GlobalContext, GlobalVolatileContext } from '../../contexts/GlobalNodeState';
import { TypeTags } from '../TypeTag';
import { OutputProps } from './props';

interface LatentImageBroadcastData {
    width: number;
    height: number;
}

export const LatentImageOutput = memo(
    ({ label, id, outputId, schemaId, useOutputData }: OutputProps) => {
        const type = useContextSelector(GlobalVolatileContext, (c) =>
            c.typeState.functions.get(id)?.outputs.get(outputId)
        );

        const { setManualOutputType } = useContext(GlobalContext);

        const outputIndex = useContextSelector(BackendContext, (c) =>
            c.schemata.get(schemaId).outputs.findIndex((o) => o.id === outputId)
        );

        const { current } = useOutputData<LatentImageBroadcastData>(outputId);
        useEffect(() => {
            if (current) {
                setManualOutputType(
                    id,
                    outputId,
                    struct('Image', {
                        width: literal(current.width),
                        height: literal(current.height),
                    })
                );
            } else {
                setManualOutputType(id, outputId, undefined);
            }
        }, [id, outputId, current, setManualOutputType]);

        return (
            <Flex
                h="full"
                minH="2rem"
                verticalAlign="middle"
                w="full"
            >
                <Spacer />
                {type && (
                    <Center
                        h="2rem"
                        verticalAlign="middle"
                    >
                        <TypeTags
                            isOptional={false}
                            type={type}
                        />
                    </Center>
                )}
                <Text
                    h="full"
                    lineHeight="2rem"
                    marginInlineEnd="0.5rem"
                    ml={1}
                    textAlign="right"
                >
                    {label}
                </Text>
            </Flex>
        );
    }
);
